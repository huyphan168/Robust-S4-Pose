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
import matplotlib.patches as mpatches

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--eval-results', type=str, default='eval_results', help='path to log file')
    parser.add_argument('-o', '--out-folder', type=str, default="plots", help='path to output folder' )
    args   = parser.parse_args()
    return args
    

def plot_exp(ax, eval_folder, to_run, shift=0.0, name = None, hatch=None, mode='line'):
    mpjpe_vals = []
    distr_names= []
    colors     = []
    
    for file, n, c in to_run:
        file_path = osp.join(eval_folder, "%s.csv" % file)
        if osp.exists(file_path):
            print(file_path)
            df = pd.read_csv(file_path)
            a_mpjpe = float(df.loc[df['action'] == 'average']['mpjpe'])
            mpjpe_vals.append(a_mpjpe)
            distr_names.append(n)
            colors.append(c)
        else:
            print("File not found %s" % file_path)
    if mode == 'line':
        ax.plot(mpjpe_vals,marker='o', label=name)
    elif mode == 'bar':
        for i in range(len(distr_names)):
            ax.bar(i + shift, mpjpe_vals[i], width = 0.2, color = colors[i], hatch=hatch)

def plot_exp_series(args, exp_series, x_ticks, xlabel="Distortion Type", ylabel="MPJPE", 
        out_file="mpjpe_plot.png", title="Plot", mode = 'line'):
    patterns= ['//','+', 'o']
    plt.figure(figsize=(14,5))
    for i, exp_id in enumerate(exp_series):
        plot_exp(plt.gca(), 
            osp.join(args.eval_results, exp_id),
            exp_series[exp_id]['exps'],
            name=exp_series[exp_id]['label'],
            shift= (i - len(exp_series)//2)*0.4 + 0.2,
            hatch= patterns[i],
            mode = mode
        )
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if mode == 'bar':
        plt.legend(handles=[mpatches.Patch(hatch=patterns[i], edgecolor='black', facecolor='white',  zorder = 1, label=exp_id) for i, exp_id in enumerate(exp_series)], loc='best')
    plt.legend(loc='best')
    plt.title(title)
    plt.grid()
    plt.savefig(osp.join(args.out_folder, out_file), bbox_inches="tight")

def plot_exp_gauss(args, exps_list, out_file="mpjpe_plot.png", 
                    distortion_parts="legs", distortion_temporal = "0.3"):
    exp_series = {}
    for i, (exp, ckpt, label) in enumerate(exps_list):
        model_name = "VideoPose3D"
        color = 'bisque'
        distortion_types  = [
            ('None', 'gray'),
            ('gauss_0.01',color),
            ('gauss_0.03',color),
            ('gauss_0.05',color),
            ('gauss_0.1' ,color),
            ('gauss_0.2' ,color),
            ('gauss_0.3' ,color),
            ('gauss_0.4' ,color),]
        list_plot = []
        for t, c in distortion_types:
            list_plot.append((
                '%s_%s_%s%s' % (
                    ckpt, t, distortion_parts,
                    "_%s" % str(distortion_temporal)
                ),
                t,
                c
            )
            )
        exp_series[exp] = {'label': label, 'exps': list_plot}
    x_ticks= [i[0] for i in distortion_types]
    title  = "[%s] Performance on distored input (%s)" % (model_name, distortion_parts)
    plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file
        )

def plot_time_distortion(args, exps_list, out_file="mpjpe_plot.png", 
                    distortion_parts="legs", distortion_type = "gauss_0.1"):
    exp_series = {}
    for i, (exp, ckpt, label) in enumerate(exps_list):
        model_name = "VideoPose3D"
        color           = 'bisque'
        distortion_ratio  = [
            # ('None', 'gray'),
            ('0.01',color),
            ('0.05',color),
            ('0.1' ,color),
            ('0.2' ,color),
            ('0.3' ,color),
            ('0.4' ,color),
            ('0.5' ,color),
            ('0.6' ,color),
            ('0.8' ,color),
            ('0.9' ,color),
            ('1.0' ,color),]
        list_plot = []
        for t, c in distortion_ratio:
            list_plot.append((
                '%s_%s_%s%s' % (
                    ckpt, distortion_type, 
                    distortion_parts,
                    "_%s" % str(t) if t != 'None' else ''
                ),
                t,
                c
            )
            )
        exp_series[exp] = {'label': label, 'exps': list_plot}
    x_ticks= [i[0] for i in distortion_ratio]
    title  = "[%s] Performance on distored input (%s)" % (model_name, distortion_parts)
    plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        xlabel  = "Temporal Distortion Ratio"
        )

def main(args):
    exps_list = [
        ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_gauss",
        "epoch_80",
        "Gauss noise $\sigma=0.1$ - 30% joints - 30% frames - gauss conf scr"),
        ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D-27-frames on H36M")
        ]
    plot_exp_gauss(args, exps_list,
        out_file = "mpjpe_plot.png"
    )
    plot_time_distortion(args, exps_list,
        out_file = "mpjpe_temporal_distortion.png",
    )
     
if __name__ == "__main__":
    args = parse_args()
    main(args)