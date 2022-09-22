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
    args   = parser.parse_args()
    return args

def main(args):
    model_name = "VideoPose3D"
    ckpt       = "pretrained_h36m_cpn"
    parts      = "arm-right" #"legs"
    TODO = [
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
    mpjpe_vals = []
    distr_names= []
    colors     = []
    plt.figure(figsize=(14,5))
    for file, name, c in TODO:
        file = file % (ckpt, parts)
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
    plt.xlabel("Input types")
    plt.ylabel("MPJPE")
    plt.title("[%s] %s performance on distored input (%s)" % (model_name, ckpt, parts))
    plt.savefig("mpjpe_plot.png", bbox_inches="tight")
     
if __name__ == "__main__":
    args = parse_args()
    main(args)