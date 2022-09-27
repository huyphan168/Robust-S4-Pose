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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--eval-results', type=str, default='eval_results', help='path to log file')
    parser.add_argument('-o', '--out-folder', type=str, default="plots", help='path to output folder' )
    args   = parser.parse_args()
    return args
    

def gen_plot(args, to_run, out_file, title, xlabel, ylabel="MPJPE"):
    mpjpe_vals = []
    distr_names= []
    colors     = []
    plt.figure(figsize=(14,5))
    for file, name, c in to_run:
        file_path = osp.join(args.eval_results, "%s.csv" % file)
        if osp.exists(file_path):
            df = pd.read_csv(file_path)
            a_mpjpe = float(df.loc[df['action'] == 'average']['mpjpe'])
            mpjpe_vals.append(a_mpjpe)
            distr_names.append(name)
            colors.append(c)

    for i in range(len(distr_names)):
        plt.bar(i, mpjpe_vals[i], width = 0.5, color = colors[i])
    plt.xticks(np.arange(len(distr_names)), distr_names)
    plt.axhline(y=mpjpe_vals[0], color='r', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(osp.join(args.out_folder, out_file), bbox_inches="tight")

def main(args):
    model_name = "VideoPose3D"
    ckpt       = "pretrained_h36m_cpn"
    parts      = "arm-right" #"legs"
    noises_lst = [
            ('%s_none_%s', 'none', 'k'),
            ('%s_gauss_0.01_%s', 'gauss 0.01', 'b'),
            ('%s_0.03_%s', 'gauss 0.03', 'b'),
            ('%s_gauss_0.05_%s', 'gauss 0.05', 'b'),
            ('%s_joint_0_%s', 'joint_0', 'g'),
            ('%s_gauss_0.08_%s', 'gauss 0.08', 'b'),
            ('%s_gauss_0.1_%s', 'gauss 0.1', 'b'),
            ('%s_gauss_0.2_%s', 'gauss 0.2', 'b'),
            ('%s_gauss_0.3_%s', 'gauss 0.3', 'b'),
            ('%s_gauss_0.4_%s', 'gauss 0.4', 'b'),
            ('%s_zero_%s', 'zero', 'g'),
            ('%s_mean_%s', 'mean', 'g')
        ]
    for i in range(len(noises_lst)):
        noises_lst[i] = (noises_lst[i][0] % (ckpt, parts) , noises_lst[i][1], noises_lst[i][2])
         
    gen_plot(args, noises_lst,
        title="[%s] %s performance on distored input (%s)" % (model_name, ckpt, parts),
        out_file="mpjpe_plot.png",
        xlabel="Distortion Type"
    )

    parts   = "legs"
    noises_lst = [
            ('%s_none_%s', 'none', 'k'),
            ('%s_gauss_0.1_%s_0.01', '1%', 'b'),
            ('%s_gauss_0.1_%s_0.05', '5%', 'b'),
            ('%s_gauss_0.1_%s_0.1', '10%', 'b'),
            ('%s_gauss_0.1_%s_0.2', '20%', 'b'),
            ('%s_gauss_0.1_%s_0.3', '30%', 'b'),
            ('%s_gauss_0.1_%s_0.4', '40%', 'b'),
            ('%s_gauss_0.1_%s_0.5', '50%', 'b'),
            ('%s_gauss_0.1_%s_1.0', '100%', 'b'),]

    for i in range(len(noises_lst)):
        noises_lst[i] = (noises_lst[i][0] % (ckpt, parts) , noises_lst[i][1], noises_lst[i][2])
         
    gen_plot(args, noises_lst,
        title="[%s] %s performance on distored input (%s)" % (model_name, ckpt, parts),
        out_file="mpjpe_dist_frames_ratio_plot.png",
        xlabel="Percentage of distorted frames"
    )
     
if __name__ == "__main__":
    args = parse_args()
    main(args)