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
from common.utils  import load_cfg_from_file
from common.camera import *
from common.h36m_dataset import h36m_cameras_intrinsic_params
from scipy.optimize import curve_fit
from common.input_distortion import InputDistortion
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import seaborn as sns
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--set', type=str, default='erase', help='path to log file')
    parser.add_argument('-o', '--out-folder', type=str, default="plots", help='path to output folder' )
    parser.add_argument('-cfg', '--cfg-file', type=str, default=None, help="path to config file")
    # Input distortion
    parser.add_argument('--train-distortion-type',    type=str, default="None")
    parser.add_argument('--train-distortion-parts',   type=str, default='None')
    parser.add_argument('--train-distortion-temporal',type=str, default='None')
    parser.add_argument('--train-gen-conf-score',     type=str, default='None')

    parser.add_argument('--test-distortion-type',     type=str, default="None")
    parser.add_argument('--test-distortion-parts',    type=str, default='None')
    parser.add_argument('--test-distortion-temporal', type=str, default='None')
    parser.add_argument('--test-gen-conf-score',      type=str, default='None')
    parser.add_argument('--drop-conf-score', action='store_true', default=False)
    # Eval filter
    parser.add_argument('--loss-ignore-parts', type=str, default='None')
    parser.add_argument('--eval-ignore-parts', type=str, default='None')

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

def compute_distance(seq1, seq2):
    return np.sqrt(((seq1 - seq2)**2).sum(axis=-1))

COCO_JOINTS = ["nose","left_eye","right_eye","left_ear","right_ear","Left Shoulder",
                "right_shoulder","left_elbow","Right Elbow","left_wrist","right_wrist",
                "Left Hip","Right Hip","left_knee","right_knee","Left Ankle","Right Ankle"]

def plot_conf_scr_heatmap(args):
    dist_kpts   = load_kpt(args.set)
    subjects    = list(dist_kpts.keys())
    actions     = list(dist_kpts[subjects[0]].keys()) 
    cameras     = [i for i in range(4)]
    # cameras     = list(dist_kpts[subjects[0]][actions[0]].keys()) 
    clean_seq = []
    dist_seq  = []
    # axs = plt.subplots(len(viz_joints),2,figsize=(16,8))
    plt.figure(figsize=(10,6))

    for subj in ['S9']:
        for act in['Walking']:
            for cam in cameras:
                conf  = dist_kpts[subj][act][cam][250:300,:,-1]
                # import ipdb; ipdb.set_trace()
                plt.imshow(conf.T, cmap='hot', interpolation='nearest')
                plt.yticks([])
                # sns.heatmap(conf.T, linewidth=0.2)
                break
    plt.xlabel("Time")
    plt.ylabel("Joint index")
    plt.savefig("plots/conf_scr_heatmap.png", bbox_inches='tight')
    # clean_seq = np.vstack(clean_seq)
    # dist_seq  = np.vstack(dist_seq)

def load_kpts(args, rsubjects='all', ractions='all', rcameras='all'):
    clean_kpts  = load_kpt('clean')
    dist_kpts   = load_kpt(args.set)
    subjects    = list(dist_kpts.keys())
    actions     = list(dist_kpts[subjects[0]].keys()) 
    cameras     = [i for i in range(4)]
    clean_seq = []
    dist_seq  = []
    for subj in subjects if rsubjects=='all' else rsubjects:
        for act in dist_kpts[subj].keys() if ractions == 'all' else ractions:
            for cam in cameras if rcameras=='all' else rcameras:
                clean = normalize(clean_kpts[subj][act][cam], cam)
                dist  = normalize(dist_kpts[subj][act][cam], cam) 
                length= min(len(clean), len(dist))
                clean_seq.append(clean[:length])
                dist_seq.append(dist[:length])
    clean_seq = np.vstack(clean_seq)
    dist_seq  = np.vstack(dist_seq)
    return clean_seq, dist_seq

def plot_err_histogram(args):
    clean_seq, dist_seq = load_kpts(args)
    joint_idx = 5
    err = compute_distance(clean_seq[...,:2], dist_seq[...,:2]).flatten()
    cnf = dist_seq[...,-1].flatten()
    # noise_msk = dist_seq[:,joint_idx,2] > 0.1
    df = pd.DataFrame()
    df = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'cnf': err[cnf<=0.4].tolist(), 'Confidence score': r'c $ \leq 0.4$'}),
        pd.DataFrame.from_dict({'cnf': err[cnf<=0.2].tolist(), 'Confidence score': r'c $ \leq 0.2$'}),
        pd.DataFrame.from_dict({'cnf': err[cnf<=0.1].tolist(), 'Confidence score': r'c $ \leq 0.1$'}),
    ])

    f, axs = plt.subplots(1, 2, figsize=(8,4))
    f.tight_layout(pad = 1.0)
    sns.histplot(
        data=df, x='cnf', hue='Confidence score', element="step", stat='probability', ax=axs[0])
    axs[0].set_xlabel(r"Error magnitude $||\tilde{x}-x||$")
    axs[0].set_ylabel(r"Probability")
    axs[0].grid()
    axs[0].set_xlim(0, 1.6)
    
    df = pd.DataFrame()
    df = pd.DataFrame.from_dict({'err': err.tolist(), 'cnf': cnf.tolist()})

    
    rrr = sns.histplot(df, x="err", y="cnf", binwidth=(.1, .05), cbar=True, stat = 'probability', ax=axs[1])
    cbar = rrr.collections[0].colorbar
    cbar.set_ticks([0, .1, .2])
    
    axs[1].set_xlabel(r"Error magnitude $||\tilde{x}-x||$")
    axs[1].set_ylabel(r"Confidence score")
    axs[1].set_xlim(0, 1.6)
    axs[1].set_ylim(0, 1.0)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(4))
    axs[1].grid(which='both')
    axs[1].ticklabel_format(style='scientific', axis='both',scilimits=[-1,1])
    axs[0].ticklabel_format(style='scientific', axis='both',scilimits=[-1,1])
    
    plt.savefig("plots/error_cnf_hist.png", bbox_inches='tight')
    plt.savefig("plots/error_cnf_hist.pdf", bbox_inches='tight')

def plot_joint_err_hist(args):
    clean_seq, dist_seq = load_kpts(args)
    tar_err = compute_distance(clean_seq[...,:2], dist_seq[...,:2])
    # ####### Plot histogram ########
    # viz_joints = np.arange(1,17)
    # n_rows, n_cols = 4, 4
    # f, axs = plt.subplots(n_rows,n_cols,figsize=(10,11))
    # f.suptitle("Histogram of the distorted 2D keypoints x-axis difference by applying %s" % args.set)
    # f.tight_layout(pad = 2.0)
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         jidx = i* n_cols + j
    #         if jidx >= len(viz_joints):
    #             continue
    #         err_seq = tar_err[:, viz_joints[jidx]]
    #         dist = (dist_seq-clean_seq)[:, viz_joints[jidx], 0]
    #         no_noise_msk = err_seq < 0.1
    #         hist, bins = np.histogram(dist[np.logical_not(no_noise_msk)], bins=100, density=True)
    #         axs[i,j].hist(bins[:-1], bins, weights=hist)
    #         axs[i,j].set_xlim(-1,1)
    #         axs[i,j].set_ylim(0,3.0)
    #         axs[i,j].set_title(COCO_JOINTS[viz_joints[jidx]])
    viz_joints = np.array([5, 8, 11, 16])
    n_rows, n_cols = 1, 4
    f, axs = plt.subplots(n_rows,n_cols,figsize=(10,3))
    f.tight_layout(pad = 2.0)

    for j in range(n_cols):
        jidx = j
        if jidx >= len(viz_joints):
            continue
        err_seq = tar_err[:, viz_joints[jidx]]
        dist = (dist_seq-clean_seq)[:, viz_joints[jidx], 0]
        no_noise_msk = err_seq < 0.1
        hist, bins = np.histogram(dist[np.logical_not(no_noise_msk)], bins=100, density=True)
        axs[j].hist(bins[:-1], bins, weights=hist)
        axs[j].set_xlim(-1,1)
        axs[j].set_ylim(0,3.0)
        axs[j].set_title(COCO_JOINTS[viz_joints[jidx]])
    axs[0].set_ylabel(args.set)
    
    plt.savefig("plots/error_histogram_joints_%s.png" % args.set, bbox_inches='tight')
    plt.savefig("plots/error_histogram_joints_%s.pdf" % args.set, bbox_inches='tight')

def plot_err_distribution(args):
    clean_seq, dist_seq = load_kpts(args)
    inp_distr = InputDistortion(args)
    augm_seq  = inp_distr.get_train_inputs(clean_seq)
    tar_err = compute_distance(clean_seq[...,:2], dist_seq[...,:2])
    sim_err = compute_distance(augm_seq[...,:2], clean_seq[...,:2])
    conf =  clean_seq[...,2]-dist_seq[...,2]
    # conf_t = np.diff(conf[:,8].flatten())
    # diff_t = np.diff(diff[:,8].flatten())
    
    ######### Plot error sequence
    # viz_joints = [7, 9, 10]
    # f, axs = plt.subplots(len(viz_joints),2,figsize=(16,8))
    # for i, j in enumerate(viz_joints):
    #     axs[i,0].plot(tar_err[:,j])
    #     axs[i,1].plot(sim_err[:,j])
    # plt.savefig("plots/error_seq.png", bbox_inches='tight')

    # # plt.scatter(tar_err[:, joint_idx], dist_seq[:,joint_idx,2])
    # # plt.scatter(tar_err[:, joint_idx], clean_seq[:,joint_idx,2])
    # plt.title("Histogram of the detected keypoints confidence score")
    # plt.savefig("plots/conf_scr_hist.png", bbox_inches='tight')

    # f, axs = plt.subplots(4,1,figsize=(10,10))
    # joint_idx = 1
    # axs[0].plot(clean_seq[:,joint_idx,0], label='x')
    # axs[0].plot(clean_seq[:,joint_idx,1], label='y')
    # axs[0].set_title("Clean Images Input")
    # axs[1].plot(clean_seq[:,joint_idx,2], label='conf')
    # axs[2].plot(dist_seq[:, joint_idx,0], label='x')
    # axs[2].plot(dist_seq[:, joint_idx,1], label='y')
    # axs[2].set_title("Erase Distorted Images Input")/mnt/data0-nfs/hthieu/repo/VideoPose3D/plots/conf_scr_hist.png/mnt/data0-nfs/hthieu/repo/VideoPose3D/plots/conf_scr_hist.png
    # axs[3].plot(dist_seq[:, joint_idx,2], label='conf')
    # for i in range(len(axs)):
    #     axs[i].legend(loc='best')
    # axs[1].plot(clean_seq[:,11,2])
    # plt.savefig("test3.png", bbox_inches='tight')
    

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
    # plot_err_distribution(args)
    # plot_conf_scr_heatmap(args)
    # plot_err_histogram(args)
    plot_joint_err_hist(args)
    
    
if __name__ == '__main__':
    args = parse_args()
    # Load arguments from file
    args = load_cfg_from_file(args, args.cfg_file)
    main(args)