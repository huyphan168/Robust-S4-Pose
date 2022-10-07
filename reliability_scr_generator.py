import argparse
from cProfile import label
import os.path as osp
from re import sub
from turtle import color
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os.path as osp
import os
import numpy as np
from common.camera import *
from common.h36m_dataset import h36m_cameras_intrinsic_params
from scipy.optimize import curve_fit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--set', type=str, default='erase', help='path to log file')
    parser.add_argument('-o', '--out-folder', type=str, default="plots", help='path to output folder' )
    parser.add_argument('--plot-cdf', action='store_true', default=False, help='plot cdf?' )
    args   = parser.parse_args()
    return args

def load_kpt(set):
    kpt_file = 'data/data_2d_h36m_hrnet_' + set + '.npz'
    kpts = np.load(kpt_file, allow_pickle=True)
    kpts = kpts['positions_2d'].item()
    return kpts

def normalize(kpts, cam):
    cam_info = h36m_cameras_intrinsic_params[cam]
    kpts[..., :2] = normalize_screen_coordinates(kpts[..., :2], w=cam_info['res_w'], h=cam_info['res_h'])
    return kpts

def process_data(args):
    clean_kpts  = load_kpt('clean')
    dist_kpts   = load_kpt(args.set)
    print(args.set)
    subjects    = list(dist_kpts.keys())
    cameras     = [i for i in range(4)]
    diff = {}
    diff_arr = []
    for subj in subjects:
        if subj not in diff:
            diff[subj] = {}
        for act in clean_kpts[subj].keys():
            if act not in diff[subj]:
                diff[subj][act] = []
            for cam in cameras:
                clean = normalize(clean_kpts[subj][act][cam], cam)
                dist  = normalize(dist_kpts[subj][act][cam] , cam) 
                length= min(len(clean), len(dist))
                # Crop clean/dist sequences
                clean = clean[:length,...,:2]
                dist  = dist [:length,...,:2] 
                e = np.sqrt(((clean-dist)**2).sum(axis=-1))
                diff_arr.append(e)
                diff[subj][act].append(e)
    
    if args.plot_cdf:
        plt.grid()
        diff_arr = np.vstack(diff_arr).flatten()
        hist, bins = np.histogram(diff_arr, bins = 1000, density = True)
        dx = bins[1] - bins[0]
        cdf= np.cumsum(hist)*dx
        plt.plot(np.log10(bins[1:]), cdf)
        plt.xlabel("Offset to the clean 2D keypoints in log scale ($log_{10}(\epsilon)$)")
        plt.ylabel("CDF")
        for r in [0.4, 0.6, 0.8, 0.85, 0.9]:
            thresh= bins[1:][np.where(cdf>r)[0][0]]
            print("Ratio = %f, thresh = %.3f" % (r,thresh))
        plt.savefig("plots/cdf_%s.png" % args.set, bbox_inches='tight')
    else:
        np.savez_compressed("data/eval_dist_h36m_hrnet_%s.npz" % args.set, eval_dist = diff)
    
    
def main(args):
    process_data(args)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)