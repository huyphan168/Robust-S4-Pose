import json
import argparse
import os.path as osp
from turtle import color
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os.path as osp
import os
import numpy as np
import cv2
from common.h36m_dataset import h36m_cameras_intrinsic_params
from common.visualization import read_video
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-folder',type=str, default='/mnt/data0-nfs/shared-datasets/human36m/processed/', help='path to log file')
    parser.add_argument('-o', '--out-folder', type=str, default="/mnt/data0-nfs/shared-datasets/human36m/processed/distorted_images", help='path to output folder')
    parser.add_argument('-s', '--subject', type=str)
    parser.add_argument('-a', '--action',  type=str)
    parser.add_argument('-c', '--camera',  type=int)
    parser.add_argument('--skip',  default=0,  type=int)
    parser.add_argument('--limit', default=30, type=int)
    args   = parser.parse_args()
    return args

def masking(frm, x, y, w, h):
    _frm = np.copy(frm)
    assert 0<=x<=1 and 0<=y<=1 and 0<=w<=1 and 0<=h<=1 
    fh, fw, _ = _frm.shape
    x1, y1 = int(x * fw), int(y * fh) 
    x2, y2 = np.clip(x1 + int(w * fw), 0, fw), np.clip(y1 + int(h * fh), 0, fh)
    _frm[y1:y2, x1:x2] = 0
    return _frm

def process_video(args):
    camera_info = h36m_cameras_intrinsic_params[args.camera]
    input_video_path = osp.join(args.root_folder, args.subject, "Videos", "%s.%s.mp4" % (args.action, camera_info['id']))
    output_video_path= osp.join(args.out_folder, args.subject,  "Videos", "%s.%s.mp4" % (args.action, camera_info['id']))
    print('Processing input video %s' % input_video_path)

    assert osp.exists(input_video_path)
    # Load video using ffmpeg
    # for f in read_video(input_video_path, skip=args.skip, limit=args.limit):
    #     all_frames.append(f)

    # Read video
    cap = cv2.VideoCapture(input_video_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 50.0, (camera_info['res_w'], camera_info['res_h']))
    os.makedirs(osp.dirname(output_video_path), exist_ok=True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out_f  = masking(frame,0.0,0.5,1.0,0.2) 
            out.write(out_f)
        else: break
    # Release everything if job is finished
    cap.release()
    out.release()
def main(args):
    process_video(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)