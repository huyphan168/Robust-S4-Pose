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
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random, load_cfg_from_file
from tqdm import tqdm
import pandas as pd
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
    args.checkpoint = osp.join(checkpoint_root, "%s-%s-a%s-b%d-dj_%s-dp_%s-df%s-lss_exc_%s-conf_%s" % (
        args.model, args.keypoints, args.architecture, args.batch_size,
        args.train_distortion_type, args.train_distortion_parts, args.train_distortion_temporal,
        args.loss_ignore_parts, args.train_gen_conf_score
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

# Prepare model
filter_widths = [int(x) for x in args.architecture.split(',')]

if args.model == "VideoPose3D":
    num_joints_in = poses_valid_2d[0].shape[-2]
    in_features   = poses_valid_2d[0].shape[-1]

    if not args.disable_optimizations and not args.dense and args.stride == 1:
        # Use optimized model for single-frame predictions
        model_pos_train = TemporalModelOptimized1f(num_joints_in, in_features, dtf.dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = TemporalModel(num_joints_in, in_features, dtf.dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                    dense=args.dense)
        
    model_pos = TemporalModel(num_joints_in, in_features, dtf.dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)
else:
    raise NotImplementedError

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

# Move models to CUDA device(s)
if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

# Load weights from pretrained models
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    model_traj = None
    if args.evaluate and 'model_traj' in checkpoint:
        if checkpoint['model_traj'] is not None:
            # Load trajectory model if it contained in the checkpoint (e.g. for inference in the wild)
            model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)
            if torch.cuda.is_available():
                model_traj = model_traj.cuda()
            model_traj.load_state_dict(checkpoint['model_traj'])
        

# Generate test data  
test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

################### TRAINING ###################
if not args.evaluate:
    # Prepare log dir
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
            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)
            # Select joints for computing losses
            predicted_3d_pos = inp_distr.get_loss_joints(predicted_3d_pos)
            inputs_3d        = inp_distr.get_loss_joints(inputs_3d)
            
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()
            optimizer.step()
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
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0
                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    # Select joints for evaluation
                    predicted_3d_pos = inp_distr.get_eval_joints(predicted_3d_pos)
                    inputs_3d        = inp_distr.get_eval_joints(inputs_3d)         
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue
                        
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    # Select joints for computing losses
                    predicted_3d_pos = inp_distr.get_loss_joints(predicted_3d_pos)
                    inputs_3d        = inp_distr.get_loss_joints(inputs_3d)

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

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
        momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        model_pos_train.set_bn_momentum(momentum)
            
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
        if losses_3d_valid[-1] * 1000 < min_loss:
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
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
            plt.close('all')

# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0

    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
    
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            
            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, dtf.joints_left + dtf.joints_right] = predicted_3d_pos[1, :, dtf.joints_right + dtf.joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
                
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0    
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]
            
            # mask 3d outputs for computing losses
            predicted_3d_pos = inp_distr.get_eval_joints(predicted_3d_pos)
            inputs_3d        = inp_distr.get_eval_joints(inputs_3d)

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')
    return e1, e2, e3, ev

################### RENDERING ###################
if args.render:
    print('Rendering...')
    input_keypoints       = dtf.keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    noisy_input_keypoints = inp_distr.get_test_inputs(input_keypoints)
    
    # Predict from clean inputs:
    prediction, ground_truth = gen_animation(input_keypoints)
    prediction_from_noise, _ = gen_animation(noisy_input_keypoints)    

    anim_output = {
        'Given Noisy Input': prediction_from_noise,
        'Given Clean Input': prediction,
        }
    if ground_truth is not None and not args.viz_no_ground_truth:
        anim_output['Ground truth'] = ground_truth
    
    input_keypoints       = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
    noisy_input_keypoints = image_coordinates(noisy_input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h']) 
    
    ignore_joints = np.where(inp_distr.get_mask(inp_distr.body_parts) == 0)[0]
    from common.visualization import dist_render_animation, render_animation
    dist_render_animation(noisy_input_keypoints, dtf.keypoints_metadata, anim_output,
                        dtf.dataset.skeleton(), dtf.dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                        limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                        input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                        input_video_skip=args.viz_skip, ignore_joints=ignore_joints)
    
else:
    print('Evaluating...')

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        actions_name = []
        for action_key in actions.keys():
            actions_name.append(action_key)
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = dtf.fetch_test_actions(actions[action_key])

            poses_2d_act = [inp_distr.get_test_inputs(i) for i in poses_2d_act]
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        # Save evaluation results to a pandas file
        m_p1 = round(np.mean(errors_p1), 2)
        m_p2 = round(np.mean(errors_p2), 2)
        m_p3 = round(np.mean(errors_p3), 2)
        m_v  = round(np.mean(errors_vel), 2)
        print('Protocol #1   (MPJPE) action-wise average:', m_p1 ,'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', m_p2, 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', m_p3, 'mm')
        print('Velocity      (MPJVE) action-wise average:', m_v, 'mm')

        actions_name.append("average")
        errors_p1.append(m_p1)
        errors_p2.append(m_p2)
        errors_p3.append(m_p3)
        errors_vel.append(m_v)

        df = pd.DataFrame({
            'action' : actions_name, 
            'mpjpe'  : errors_p1,
            'p-mpjpe': errors_p2,
            'n-mpjpe': errors_p3,
            'mpjve'  : errors_vel
            })
        fold_name = osp.join("eval_results", osp.basename(args.checkpoint))
        os.makedirs(fold_name, exist_ok=True)
        file_name = osp.join(fold_name, "%s_%s_%s%s.csv" % (
            args.evaluate.split('.')[0],
            args.test_distortion_type,
            args.test_distortion_parts,
            "_%s" % args.test_distortion_temporal if args.test_distortion_temporal != 'None' else '' 
        ))
        df.round(2).to_csv(file_name, index=False)
        
    if not args.by_subject:
        run_evaluation(dtf.all_actions, dtf.action_filter)
    else:
        for subject in dtf.all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(dtf.all_actions_by_subject[subject], dtf.action_filter)
            print('')