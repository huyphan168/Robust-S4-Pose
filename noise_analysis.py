import argparse
from cProfile import label
import os.path as osp
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

def laplace_func(x, mu, b):
    return 1/(2*b)*np.exp(-np.abs(x-mu)/b)

def plot_err_distribution(args):
    clean_kpts  = load_kpt('clean')
    dist_kpts   = load_kpt(args.set)
    subjects    = list(dist_kpts.keys())
    actions     = list(dist_kpts[subjects[0]].keys()) 
    cameras     = [i for i in range(4)]
    # cameras     = list(dist_kpts[subjects[0]][actions[0]].keys()) 
    # for subject in ['S9']
    diff = []
    conf = []
    clean_seq = []
    dist_seq  = []
    for subj in ['S9']: #subjects:
        for act in ['Greeting']: # clean_kpts[subj].keys():
            for cam in cameras:
                clean = normalize(clean_kpts[subj][act][cam], cam)
                dist  = normalize(dist_kpts[subj][act][cam], cam) 
                length= min(len(clean), len(dist))
                e = clean[:length,...,:2] - dist[:length, ...,:2] 
                diff.append(e)
                conf.append(dist[:length,...,-1] - clean[:length,...,-1])
                clean_seq.append(clean[:length])
                dist_seq.append(dist[:length])
                break
                # import ipdb; ipdb.set_trace()
    diff = np.vstack(diff)
    diff = np.sqrt((diff**2).sum(axis=-1))
    conf = np.vstack(conf)
    clean_seq = np.vstack(clean_seq)
    dist_seq  = np.vstack(dist_seq)
    conf_t = np.diff(conf[:,8].flatten())
    diff_t = np.diff(diff[:,8].flatten())
    
    # f, axs = plt.subplots(2,1,figsize=(10,8))
    
    # axs[0].plot(conf_t)
    # axs[0].plot(diff_t)
    # axs[1].plot(np.abs(conf_t) * np.abs(diff_t))
    # plt.savefig("test2.png", bbox_inches='tight')

    f, axs = plt.subplots(4,1,figsize=(10,10))
    joint_idx = 1
    axs[0].plot(clean_seq[:,joint_idx,0], label='x')
    axs[0].plot(clean_seq[:,joint_idx,1], label='y')
    axs[0].set_title("Clean Images Input")
    axs[1].plot(clean_seq[:,joint_idx,2], label='conf')
    axs[2].plot(dist_seq[:, joint_idx,0], label='x')
    axs[2].plot(dist_seq[:, joint_idx,1], label='y')
    axs[2].set_title("Erase Distorted Images Input")
    axs[3].plot(dist_seq[:, joint_idx,2], label='conf')
    for i in range(len(axs)):
        axs[i].legend(loc='best')
    # axs[1].plot(clean_seq[:,11,2])
    plt.savefig("test3.png", bbox_inches='tight')
    
    # plt.figure(figsize=(10,8))
    # dist_diff = np.sqrt((diff**2).sum(axis=-1)).flatten()
    # conf_diff = conf.flatten()
    # import ipdb; ipdb.set_trace()
    # plt.scatter(np.diff(dist_diff),np.diff(conf_diff))
    # diff = np.clip(diff, -0.5,0.5)
    # diff = diff[:,10,1]
    # print(diff.shape)
    # plt.figure(figsize=(10,8))
    # # Plot histogram
    # hist, bins = np.histogram(diff.flatten(), bins=1000, density=True)
    # plt.hist(bins[:-1], bins, weights=hist)
    # # Curve fitting
    # p0 = [0., .1]
    # coeff, var_matrix = curve_fit(laplace_func, bins[:-1], hist, p0=p0)
    # x = np.linspace(bins.min(), bins.max())
    # y_fit = laplace_func(x, *coeff)
    # print(coeff)
    # plt.plot(x, y_fit,color='red')
    
def main(args):
    plot_err_distribution(args)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)