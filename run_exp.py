# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from email.mime import image
import os
from common.input_distortion import InputDistortion
import numpy as np
from common.arguments import parse_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import logging
import datetime
from common.camera import *
from common.loss   import *
from common.data   import *
from common.render import *
from common.model  import *
from common.evaluate import evaluate, run_evaluation
from common.model_factory import get_model, initialize_model
from common.generators import ChunkedGenerator, UnchunkedGenerator, EvaluateGenerator
from time import time
from common.utils import deterministic_random, load_cfg_from_file, set_momentum
from tqdm import tqdm
import os.path as osp
# Parse argument
args = parse_args()
# Load arguments from file
args = load_cfg_from_file(args, args.cfg_file)
print(args, "\n")
# Set GPU device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
# Making the experiments reproducible
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Create checkpoint directory
if args.checkpoint == "auto":
    checkpoint_root = "checkpoint"
    args.checkpoint = osp.join(checkpoint_root, "%s-%s-a%s-b%d-dj_%s-dp_%s-df%s-lss_exc_%s-conf_%s%s%s" % (
        args.model, args.keypoints, args.architecture, args.batch_size,
        args.train_distortion_type, args.train_distortion_parts, args.train_distortion_temporal,
        args.loss_ignore_parts, 
        'det' if ('hrnet' in args.keypoints) and  args.drop_conf_score == False else args.train_gen_conf_score,
        '_' + args.loss if args.loss != 'mpjpe' else '',
        '_smthconf' if args.smooth_conf_score else ''
        ))
print("Saving checkpoint to ", args.checkpoint, "\n")
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

# Prepare input distortion instance
inp_distr = InputDistortion(args)
# Prepare dataset
dtf = DataFetcher(args) 
# Fetch validation dataset
cameras_valid, poses_valid, poses_valid_2d = dtf.fetch_test()
# Apply distortion on validation data input
poses_valid_2d = [inp_distr.get_test_inputs(i) for i in poses_valid_2d]

model_pos, model_pos_train, checkpoint = initialize_model(args,
    num_joints_in  = poses_valid_2d[0].shape[-2] if len(poses_valid_2d) > 0 else 17, 
    num_joints_out = dtf.dataset.skeleton().num_joints(),
    in_features    =  poses_valid_2d[0].shape[-1] if len(poses_valid_2d) > 0 else 3 if args.drop_conf_score == False else 2,
    chk_filename   = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
)

model_traj = None
# Generate test data
if hasattr(model_pos, 'receptive_field'):
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field-1)//2 # Padding on each side
else:
    receptive_field = 1
    for i in args.architecture.split(','):
        receptive_field *= int(i)
        pad = (receptive_field-1)//2
print('INFO: Receptive field: {} frames'.format(receptive_field))

if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

if args.test_fixed_size_input:
    test_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_valid, poses_valid, poses_valid_2d, args.stride,
                                  pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation, shuffle=False,
                                  kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)
else:
    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

################### TRAINING ###################
if not args.evaluate:
    # Prepare log dir
    import wandb
    wandb.init(project="robustvp3d", name=args.wandb_name, config=args)

    logging.basicConfig(filename= #"test.log",
    osp.join(args.checkpoint, "%s.log" % datetime.datetime.now()),
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    # Save args to log file
    logging.info(args)

    # Fetch train data
    cameras_train, poses_train, poses_train_2d = dtf.fetch_train()
    # Apply distortion on input training data 
    # from matplotlib import pyplot as plt
    # plt.plot(poses_train_2d[0][:,11,0])
    poses_train_2d = [inp_distr.get_train_inputs(i) for i in poses_train_2d]
    # Prepare optimizers
    lr = args.learning_rate
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)    
    lr_decay = args.lr_decay
    
    min_loss = 1e4
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        
        lr = checkpoint['lr']
            
    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()
        # Regular supervised scenario
        for _, batch_3d, batch_2d in tqdm(train_generator.next_epoch(), total=train_generator.num_batches):
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()
            # inputs_2d = inp_distr.smooth_conf_scr(inputs_2d)
            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)
            # Select joints for computing losses
            predicted_3d_pos = inp_distr.get_loss_joints(predicted_3d_pos)
            inputs_3d        = inp_distr.get_loss_joints(inputs_3d)
            
            if   args.loss == 'mpjpe':
                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            elif args.loss == 'l1':
                loss_3d_pos = L1_loss(predicted_3d_pos, inputs_3d)
            elif args.loss == "conf_mpjpe":
                loss_3d_pos = conf_mpjpe(predicted_3d_pos, inputs_3d, inputs_2d[...,-1])
            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()
            optimizer.step()
            wandb.log({"loss_3d_pos": loss_3d_pos.item()*1000})

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()
            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0
            
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if args.smooth_conf_score == True or "PoseFormer" in args.model:
                        inputs_2d, inputs_3d = eval_data_prepare(inputs_2d, inputs_3d, receptive_field)
                    
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0
                   
                    # Smooth conf. score (if enabled)
                    # inputs_2d = inp_distr.smooth_conf_scr(inputs_2d)
                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    # Select joints for evaluation
                    predicted_3d_pos = inp_distr.get_eval_joints(predicted_3d_pos)
                    inputs_3d        = inp_distr.get_eval_joints(inputs_3d)         
                    try:
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    except:
                        import ipdb; ipdb.set_trace()
                        loss_3d_pos = mpjpe(predicted_3d_pos[:, :inputs_3d.shape[1]], inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 1 # SHOULD BE ZERO!!
                # for cam, batch, batch_2d in train_generator_eval.next_epoch():
                #     if batch_2d.shape[1] == 0:
                #         # This can only happen when downsampling the dataset
                #         continue
                        
                #     inputs_3d = torch.from_numpy(batch.astype('float32'))
                #     inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                #     if args.smooth_conf_score == True or "PoseFormer" in args.model:
                #         inputs_2d, inputs_3d = eval_data_prepare(inputs_2d, inputs_3d, receptive_field)

                #     if torch.cuda.is_available():
                #         inputs_3d = inputs_3d.cuda()
                #         inputs_2d = inputs_2d.cuda()
                #     inputs_traj = inputs_3d[:, :, :1].clone()
                #     inputs_3d[:, :, 0] = 0
                #     # Smooth conf. score (if enabled)
                #     # inputs_2d = inp_distr.smooth_conf_scr(inputs_2d)
                #     # Compute 3D poses
                #     predicted_3d_pos = model_pos(inputs_2d)
                #     # Select joints for computing losses
                #     predicted_3d_pos = inp_distr.get_loss_joints(predicted_3d_pos)
                #     inputs_3d        = inp_distr.get_loss_joints(inputs_3d)

                #     loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                #     epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                #     N += inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)
                wandb.log({"loss_3d_pos_valid": losses_3d_valid[-1]*1000})
                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0

        elapsed = (time() - start_time)/60
        
        if args.no_eval:
            logging.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
        else:
            logging.info('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1]  *1000))
        
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        
        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # if args.parallel:
        #     set_momentum(model_pos_train, momentum)
        # else:
        #     model_pos_train.set_bn_momentum(momentum)
            
        # Save checkpoint if necessary
        def save_ckpt(chk_path):
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': None,
                'random_state_semi':  None,
            }, chk_path)
        if epoch % args.checkpoint_frequency == 0:
            save_ckpt(os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch)))
        if len(losses_3d_valid) > 0:
            if  losses_3d_valid[-1] * 1000 < min_loss:
                save_ckpt(os.path.join(args.checkpoint, 'best.bin'.format(epoch)))
                min_loss = losses_3d_valid[-1] * 1000
        
        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (mm)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
            plt.close('all')



def gen_animation(model_pos, input_keypoints):
    ground_truth = None
    if args.viz_subject in dtf.dataset.subjects() and args.viz_action in dtf.dataset[args.viz_subject]:
        if 'positions_3d' in dtf.dataset[args.viz_subject][args.viz_action]:
            ground_truth = dtf.dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)
    
    # First set of perdiction
    prediction = evaluate(model_pos, gen, inp_distr, dtf, return_predictions=True)

    if model_traj is not None and ground_truth is None:
        raise NotImplementedError
    
    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)
    
    if ground_truth is not None:
        # Reapply trajectory
        trajectory = ground_truth[:, :1]
        ground_truth[:, 1:] += trajectory
        prediction += trajectory
    
    # Invert camera transformation
    cam = dtf.dataset.cameras()[args.viz_subject][args.viz_camera]
    if ground_truth is not None:
        prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
        ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
    else:
        # If the ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        for subject in dtf.dataset.cameras():
            if 'orientation' in dtf.dataset.cameras()[subject][args.viz_camera]:
                rot = dtf.dataset.cameras()[subject][args.viz_camera]['orientation']
                break
        prediction = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    return prediction, ground_truth

################### RENDERING ###################
if args.render:
    print('Rendering...')
    input_keypoints       = dtf.keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    noisy_input_keypoints = inp_distr.get_test_inputs(input_keypoints)
    # Predict from clean inputs:
    # prediction, ground_truth = gen_animation(input_keypoints)
    prediction_source_1, ground_truth = gen_animation(model_pos, noisy_input_keypoints)    
    # Second set of prediction
    model_pos_2, _, _ = initialize_model(args,
        num_joints_in  = poses_valid_2d[0].shape[-2] if len(poses_valid_2d) > 0 else 17, 
        num_joints_out = dtf.dataset.skeleton().num_joints(),
        in_features    =  poses_valid_2d[0].shape[-1] if len(poses_valid_2d) > 0 else 3 if args.drop_conf_score == False else 2,
        chk_filename   = os.path.join(args.checkpoint_2, args.resume if args.resume else args.evaluate)
    )
    prediction_source_2, _ = gen_animation(model_pos_2, noisy_input_keypoints) 

    anim_output = {
        'Model 1': prediction_source_1,
        'Model 2': prediction_source_2
        }
    if ground_truth is not None and not args.viz_no_ground_truth:
        anim_output['Ground truth'] = ground_truth
    from common.h36m_dataset import h36m_cameras_intrinsic_params
    cam = h36m_cameras_intrinsic_params[args.viz_camera]

    # input_keypoints       = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
    noisy_input_keypoints = image_coordinates(noisy_input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h']) 
    ignore_joints = None
    # ignore_joints = np.where(inp_distr.get_mask(inp_distr.body_parts) == 0)[0]
    from common.visualization import dist_render_animation, render_animation
    
    dist_render_animation(noisy_input_keypoints, dtf.keypoints_metadata, anim_output,
                        dtf.dataset.skeleton(), dtf.dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                        limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                        input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                        input_video_skip=args.viz_skip, ignore_joints=ignore_joints)
else:
    print('Evaluating...')
        
    if not args.by_subject:
        run_evaluation(model_pos, actions=dtf.all_actions, args=args, dtf=dtf, inp_distr=inp_distr, action_filter=dtf.action_filter)
    else:
        for subject in dtf.all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(model_pos, actions=dtf.all_actions_by_subject[subject], args=args, dtf=dtf, inp_distr=inp_distr, action_filter=dtf.action_filter)
            print('')