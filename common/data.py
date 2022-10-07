from common.camera import *
from common.utils import deterministic_random

def load_data(args, all_subjects = None):
    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path, include_subjects=all_subjects)

    elif args.dataset.startswith('humaneva'):
        from common.humaneva_dataset import HumanEvaDataset
        dataset = HumanEvaDataset(dataset_path)
    elif args.dataset.startswith('custom'):
        from common.custom_dataset import CustomDataset
        dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    else:
        raise KeyError('Invalid dataset')
    
    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]   
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
    return dataset

cam_map = {
    0: '54138969',
    1: '55011271',
    2: '58860488',
    3: '60457274',
}

class DataFetcher:
    def __init__(self, args) -> None:
        # Prepare training/testing subjects
        subjects_train = [s for s in args.subjects_train.split(',') if s!= '' ]
        if not args.render:
            subjects_test = [s for s in args.subjects_test.split(',') if s!= '' ]
        else:
            subjects_test = [args.viz_subject]

        # Loading 2D keypoints detections
        dataset = load_data(args, all_subjects = subjects_train + subjects_test)
        kpt_file  = 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'
        keypoints = np.load(kpt_file, allow_pickle=True)
        print('Loading 2D detections from %s' % kpt_file)
        keypoints_metadata = keypoints['metadata'].item()
        keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()
        
        # Loading 2D evaluation mask
        if 'file' in args.eval_ignore_parts:
            msk_npy_file = "data/eval_dist_%s_%s.npz" % (args.dataset,  args.keypoints)
            eval_dist    = np.load(msk_npy_file, allow_pickle= True)
            self.eval_dist    = eval_dist['eval_dist'].item()   
            self.eval_thr = float(args.eval_ignore_parts.split('_')[1])

        # Check detected keypoints:
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                for cam_idx in range(len(keypoints[subject][action])):
                    if keypoints[subject][action][cam_idx] is None:
                        print("%s/%s.%s.mp4" %(subject, action, cam_map[cam_idx]))
                    else:
                        mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                        if keypoints[subject][action][cam_idx].shape[0] < mocap_length:
                            print("%s/%s.%s.mp4" %(subject, action, cam_map[cam_idx]))
                            print("[ERROR] Video %s-%s-%d has %d frames, expected at least %d" % (
                                subject, action, cam_idx,
                                keypoints[subject][action][cam_idx].shape[0], mocap_length
                                ))
        for subject in dataset.subjects():
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
                if 'positions_3d' not in dataset[subject][action]:
                    continue
                    
                for cam_idx in range(len(keypoints[subject][action])):
                    
                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                    
                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                        if 'file' in args.eval_ignore_parts:
                            self.eval_dist[subject][action][cam_idx] = self.eval_dist[subject][action][cam_idx][:mocap_length]
                assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    if kps is None:
                        print('2D keypoints of subject %s action %s is None' % (subject, action))
                        continue
                    # Normalize camera frame
                    cam = dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps

        action_filter = None if args.actions == '*' else args.actions.split(',')
        if action_filter is not None:
            print('Selected actions:', action_filter)

        # Fetch test actions
        all_actions = {}
        all_actions_by_subject = {}
        for subject in subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))
        
        # Save processed keypoints/dataset
        self.args = args
        self.keypoints = keypoints
        self.downsample= args.downsample
        self.dataset   = dataset
        self.subjects_train = subjects_train
        self.subjects_test  = subjects_test
        self.action_filter  = action_filter
        self.keypoints_metadata = keypoints_metadata
        self.all_actions = all_actions
        self.all_actions_by_subject = all_actions_by_subject
        
    def __fetch(self, keypoints, subjects, action_filter=None, subset=1, parse_3d_poses=True, fetch_test=False):
        stride   = self.downsample
        dataset  = self.dataset
        out_poses_3d = []
        out_poses_2d = []
        out_eval_msk_3d  = []
        out_camera_params = []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                    
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])
                    
                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])
                    
                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        ps3d = poses_3d[i]
                        if fetch_test and 'file' in self.args.eval_ignore_parts:
                            eval_msk_3d = self.eval_dist[subject][action][i] < self.eval_thr
                            ps3d = np.concatenate([ps3d, eval_msk_3d[...,None]],axis=-1)
                        out_poses_3d.append(ps3d)
        
        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None
        
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        return out_camera_params, out_poses_3d, out_poses_2d
    

    def fetch_test(self):
        return self.__fetch(self.keypoints, self.subjects_test, self.action_filter, fetch_test=True)
    
    def fetch_train(self):
        return self.__fetch(
            self.keypoints, 
            self.subjects_train, self.action_filter, subset=self.args.subset)
    
    def fetch_test_actions(self, actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = self.keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = self.dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                ps3d = poses_3d[i]
                if 'file' in self.args.eval_ignore_parts:
                    eval_msk_3d = self.eval_dist[subject][action][i] < self.eval_thr
                    ps3d = np.concatenate([ps3d, eval_msk_3d[...,None]],axis=-1)
                out_poses_3d.append(ps3d)
                
        stride = self.args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_poses_3d, out_poses_2d

def eval_data_prepare(inputs_2d, inputs_3d, receptive_field=27):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p
