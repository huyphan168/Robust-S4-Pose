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
    

def plot_exp(ax, eval_folder, to_run, shift=0.0, name = None, hatch=None, mode='line', metrics='all', color = 'b'):
    
    distr_names= []
    colors = []
    metrics_styles     = ['-', '--', '.']
    if type(metrics) == str:
        metrics = [metrics]
    
    for mi, m in enumerate(metrics):
        mpjpe_vals = []
        for si, (file, n, c) in enumerate(to_run):
            file_path = osp.join(eval_folder, "%s%s.csv" % (file,
                "_file_%s" % m if m != 'all' else ''
            ))
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
            ax.plot(mpjpe_vals,marker='o', label=name if m=='all' else "[$MPJPE_{\leq%s}$] %s" % (m, name),
                linestyle = metrics_styles[mi], color=color
            )
        elif mode == 'bar':
            for i in range(len(distr_names)):
                ax.bar(i + shift, mpjpe_vals[i], width = 0.2, color = colors[i], hatch=hatch)

def plot_exp_series(args, exp_series, x_ticks, xlabel="Distortion Type", ylabel="MPJPE (mm)", 
        out_file="mpjpe_plot.png", title="Plot", mode = 'line', metrics = 'all'):
    patterns= ['//','+', 'o', '-', '\\', '.', '*', "\\"]
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(14,8))
    for i, exp_id in enumerate(exp_series):
        plot_exp(plt.gca(), 
            osp.join(args.eval_results, exp_id),
            exp_series[exp_id]['exps'],
            name=exp_series[exp_id]['label'],
            shift= (i - len(exp_series)//2)*0.4 + 0.2,
            hatch= patterns[i],
            mode = mode,
            metrics = metrics,
            color=colors[i]
        )
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if mode == 'bar':
        plt.legend(handles=[mpatches.Patch(hatch=patterns[i], edgecolor='black', facecolor='white',  zorder = 1, label=exp_id) for i, exp_id in enumerate(exp_series)], loc='best')
    else:
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
    title  = "[%s] Performance on distored %s keypoints" % (model_name, distortion_parts)
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
    title  = "[%s] Performance on distored %s keypoints" % (model_name, distortion_parts)
    plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        xlabel  = "Temporal Distortion Ratio",
        )

def plot_img_distortion(args, exps_list, out_file="mpjpe_img_distortion.png", 
                dist_thr = None):
    exp_series = {}
    for i, (exp, ckpt, label) in enumerate(exps_list):
        model_name = "VideoPose3D"
        color = 'bisque'
        distortion_types  = [
            ('clean', 'gray'),
            # ('brightness',color),
            ('gaussian_noise',color),
            ('impulse_noise',color),
            ('temporal' ,color),
            ('erase' ,color),
            ('crop' ,color),
            ('motion_blur' ,color),
            # ('fog' ,color),
        ]
        list_plot = []
        for t, c in distortion_types:
            list_plot.append((
                '%s_hrnet_%s_None_None%s' % (
                    ckpt, t,
                    "_file_%s" % dist_thr if str(dist_thr) is not None else '' 
                ),
                t,
                c
            )
            )
        exp_series[exp] = {'label': label, 'exps': list_plot}
    x_ticks= [i[0] for i in distortion_types]
    title  = "[%s] Performance on distored images input" % (model_name)
    plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        # mode='bar'
        )

def plot_mpjpe_rel_mpjpe(args, exps_list, out_file="rel-mpjpe-compare.png", metrics =['0.1']):
    exp_series = {}
    color = 'bisque'
    distortion_types  = [
        ('clean', 'gray'),
        # ('brightness',color),
        ('gaussian_noise',color),
        ('impulse_noise',color),
        ('temporal' ,color),
        ('erase' ,color),
        ('crop' ,color),
        ('motion_blur' ,color),
        # ('fog' ,color),
    ]
    for i, (exp, ckpt, label) in enumerate(exps_list):
        model_name = "VideoPose3D"  
        list_plot     = []
        list_plot_all = []
        for t, c in distortion_types:
            list_plot_all.append((
                '%s_hrnet_%s_None_None' % (
                    ckpt, t, 
                ), t, c
            )
            )
        exp_series[exp] = {'label': label, 'exps': list_plot_all}
    x_ticks= [i[0] for i in distortion_types]
    title  = "[%s] Performance on distored images input" % (model_name)
    plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        metrics = metrics
        )

def plot_dist_kpts(args):
    exps_list = [
        # Original VP3D
        ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN cpn keypoints"),
        
        ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.05-dp_rand_0.2-df0.2-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on dist-H36M-kpts (Gauss $\sigma=0.05$ - 20% joints - 20% frames)"),
        
        ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on dist-H36M-kpts (Gauss $\sigma=0.1$ - 30% joints - 30% frames)"),

        # ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.5-dp_rand_0.5-df1.0-lss_exc_None-conf_None",
        # "epoch_80",
        # "VP3D on dist-H36M-kpts (Gauss $\sigma=0.3$ - 50% joints- 50% frames)"),       
        ]

    # 
    plot_exp_gauss(args, exps_list,
        out_file = "mpjpe_dist_kpts_noise_level.png"
    )
    plot_time_distortion(args, exps_list,
        out_file = "mpjpe_dist_kpts_temp_distortion.png",
    )

def plot_dist_imgs(args):
    exps_list = [
        ("VideoPose3D-detectron_coco_h36m",
        "pretrained_h36m_detectron_coco",
        "Original pre-trained (243f) model"
        ),
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on mix-aug-train-hrnet"),

        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "VP3D on mix-aug-train-hrnet with det. CONF SCR input"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN hrnet det."),

        # ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        # "epoch_80",
        # "VP3D on CLEAN hrnet det. with CONF SCR input"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN + noise $\sigma=0.3$ hrnet det."),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN + noise $\sigma=0.3$ hrnet det."),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_laplace_0.02-dp_rand_0.3-df0.3-lss_exc_None-conf_det",
        "epoch_80",
        "VP3D on CLEAN + LAPLACE noise $scl=0.02$ hrnet det."), 
        
    ]
    plot_img_distortion(args, exps_list)

def plot_mpjpe_at_t_img_distortion(args):
    exps_list = [
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN hrnet det"),
        
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on MIX-AUG hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img.png", metrics=[0.1, 0.01])

    exps_list = [
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN hrnet det"),
        
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on MIX-AUG hrnet det"),
        
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.05-dp_rand_0.2-df0.2-lss_exc_None-conf_None",
        "epoch_80",
        "P3D on CLEAN + GAUSS noise $\sigma=0.05$ hrnet det"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_None",
        "epoch_80",
        "P3D on CLEAN + GAUSS noise $\sigma=0.1$ hrnet det"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "P3D on CLEAN + GAUSS noise $\sigma=0.3$ hrnet det"),
        # VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det
        # ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_impulse_0.2-dp_rand_0.3-df0.2-lss_exc_None-conf_None",
        # "epoch_80",
        # "P3D on CLEAN + IMPULSE noise $p=0.2$ hrnet det"),

        # ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_impulse_0.4-dp_rand_0.3-df0.3-lss_exc_None-conf_None",
        # "epoch_80",
        # "P3D on CLEAN + IMPULSE noise $p=0.4$ hrnet det"),

        # ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_impulse_0.6-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        # "epoch_80",
        # "P3D on CLEAN + IMPULSE noise $p=0.6$ hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img_gauss.png", metrics=[0.1])
    # plot_mpjpe_rel_mpjpe(args, exps_list, out_file="reliable-mpjpe.png", dist_thr=0.05)

def plot_conf_scr_learning(args):
    exps_list = [
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN hrnet det"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "VP3D on CLEAN + CONF SCORE hrnet det"),
        
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det_smthconf",
        "epoch_80",
        "VP3D on CLEAN + CONF SCORE NORMALIZED hrnet det"),

        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on MIX-AUG hrnet det"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det_smthconf",
        "epoch_80",
        "VP3D on MIX-AUG + CONF SCORE NORMALIZED hrnet det"),

        # ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        # "epoch_80",
        # "VP3D on MIX-AUG + CONF SCORE hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img_conf_scr.png", metrics=[0.1])

    # exps_list = [
    #     # Original VP3D
    #     ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
    #     "epoch_80",
    #     "VP3D on CLEAN cpn keypoints"),
    #     ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
    #     "epoch_80",
    #     "VP3D on CLEAN cpn keypoints"),
        
    #     ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.05-dp_rand_0.2-df0.2-lss_exc_None-conf_None",
    #     "epoch_80",
    #     "VP3D on dist-H36M-kpts (Gauss $\sigma=0.05$ - 20% joints - 20% frames)"),
        
    #     ("VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_None",
    #     "epoch_80",
    #     "VP3D on dist-H36M-kpts (Gauss $\sigma=0.1$ - 30% joints - 30% frames)"),
    # ]

    # plot_exp_gauss(args, exps_list,
    #     out_file = "mpjpe_dist_kpts_noise_level.png")

def main(args):
    # plot_dist_kpts(args)    
    # plot_dist_imgs(args)
    # plot_mpjpe_at_t_img_distortion(args)
    plot_conf_scr_learning(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)