
from optparse import check_builtin
from tabnanny import check
from common.camera import *
from common.h36m_dataset import Human36mDataset
# all_subjects=['S9', 'S11']
# check_list = ['erase', 'temporal', 'fog', 'gaussian_noise', 'impulse_noise', 'brightness', 'motion_blur', 'crop']
all_subjects =['S1','S5','S6','S7','S8','S9', 'S11']
check_list =  ['clean']

dataset = Human36mDataset('data/data_3d_h36m.npz', include_subjects=all_subjects)
cam_map = {
    0: '54138969',
    1: '55011271',
    2: '58860488',
    3: '60457274',
}

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



for kp in check_list:
    kpt_file  = 'data/data_2d_h36m_litehrnet_' + kp + '.npz'
    keypoints = np.load(kpt_file, allow_pickle=True)
    print('Loading 2D detections from %s' % kpt_file)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints = keypoints['positions_2d'].item()

    ok = True
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            for cam_idx in range(len(keypoints[subject][action])):
                if keypoints[subject][action][cam_idx] is None:
                    ok = False
                    print("%s/%s.%s.mp4" %(subject, action, cam_map[cam_idx]))
                else:
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    if keypoints[subject][action][cam_idx].shape[0] < mocap_length:
                        print("%s/%s.%s.mp4" %(subject, action, cam_map[cam_idx]))
                        # print("[ERROR] Video %s-%s-%d has %d frames, expected at least %d" % (
                        #     subject, action, cam_idx,
                        #     keypoints[subject][action][cam_idx].shape[0], mocap_length
                        #     ))
                        ok = False
    if ok:
        print(kp, " is okay")
