import torch
import numpy as np
import pandas as pd
import os.path as osp
from common.data import eval_data_prepare
from common.loss import mean_velocity_error, mpjpe, p_mpjpe, n_mpjpe
from common.generators import UnchunkedGenerator, EvaluateGenerator
# Evaluate
def evaluate(model_pos, test_generator, inp_distr, dtf, model_traj = None, action=None, return_predictions=False, use_trajectory_model=False):
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
        for (_, batch, batch_2d, _) in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if batch is not None:
                inputs_3d = torch.from_numpy(batch.astype('float32'))
            # import ipdb; ipdb.set_trace()
            # if args.smooth_conf_score == True or "PoseFormer" in args.model:
            #     inputs_2d, inputs_3d = eval_data_prepare(inputs_2d, inputs_3d, receptive_field=model_pos.receptive_field())
                # inputs_2d, inputs_3d = eval_data_prepare(inputs_2d[0, None])
                
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                if batch is not None:
                    inputs_3d = inputs_3d.cuda()
            
            # Smooth conf. score (if enabled)
            # inputs_2d = inp_distr.smooth_conf_scr(inputs_2d)
            
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
                
            inputs_3d[:, :, 0] = 0    
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]
            
            # masking 3d outputs for computing losses
            if inputs_3d.shape[-1] == 4:
                eval_msk  = inputs_3d[...,3].type(torch.bool)
                inputs_3d = inputs_3d[...,:3]
                n_frames  = eval_msk.any(dim=-1).sum().item()
            else:
                predicted_3d_pos = inp_distr.get_eval_joints(predicted_3d_pos)
                inputs_3d        = inp_distr.get_eval_joints(inputs_3d)
                eval_msk         = None
                n_frames         = inputs_3d.shape[0]*inputs_3d.shape[1]
            error = mpjpe(predicted_3d_pos, inputs_3d, eval_msk=eval_msk)
            epoch_loss_3d_pos_scale +=  n_frames * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += n_frames * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            epoch_loss_3d_pos_procrustes += n_frames * p_mpjpe(predicted_3d_pos, inputs)
            # Compute velocity error
            epoch_loss_3d_vel += n_frames * mean_velocity_error(predicted_3d_pos, inputs)

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

def run_evaluation(model_pos, actions, args,  dtf, inp_distr,  action_filter=None):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []
    actions_name = []

    # pad = (model_pos.receptive_field() - 1) // 2 # Padding on each side
    if hasattr(model_pos, 'receptive_field'):
        receptive_field = model_pos.receptive_field()
        pad = (receptive_field-1)//2 # Padding on each side
    else:
        receptive_field = 1
        for i in args.architecture.split(','):
            receptive_field *= int(i)
            pad = (receptive_field-1)//2
    causal_shift = pad if args.causal else 0
    for action_key in actions.keys():
        
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue
        actions_name.append(action_key)
        poses_act, poses_2d_act = dtf.fetch_test_actions(actions[action_key])
        poses_2d_act = [inp_distr.get_test_inputs(i) for i in poses_2d_act]
        if args.test_fixed_size_input:
            gen = EvaluateGenerator(1024, None, poses_act, poses_2d_act, args.stride,
                                    pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                    shuffle=False,
                                    kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left,
                                    joints_right=dtf.joints_right)
        else:
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                    pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                    kps_left=dtf.kps_left, kps_right=dtf.kps_right, joints_left=dtf.joints_left, joints_right=dtf.joints_right)

        e1, e2, e3, ev = evaluate(model_pos, gen, inp_distr, dtf, action=action_key)
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
    import os
    os.makedirs(fold_name, exist_ok=True)
    file_name = osp.join(fold_name, "%s%s_%s_%s_%s%s.csv" % (
        args.evaluate.split('.')[0],
        "_%s" % args.keypoints if args.keypoints != 'cpn_ft_h36m_dbb' else '',
        args.test_distortion_type,
        args.test_distortion_parts,
        args.eval_ignore_parts,
        "_%s" % args.test_distortion_temporal if args.test_distortion_temporal != 'None' else '' 
    ))
    df.round(2).to_csv(file_name, index=False)