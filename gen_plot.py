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
import tikzplotlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--eval-results', type=str, default='eval_results', help='path to log file')
    parser.add_argument('-o', '--out-folder', type=str, default="plots", help='path to output folder' )
    args   = parser.parse_args()
    return args

markers = ["o", "x", "s", "D", ">"]

def plot_exp(ax, eval_folder, to_run, shift=0.0, name = None, hatch=None, mode='line', metrics='all', color = 'b', marker='o', linestyle=None):
    
    distr_names= []
    colors = []
    metrics_styles     = ['-', '--', '.']
    if type(metrics) == str:
        metrics = [metrics]
    
    for mi, m in enumerate(metrics):
        mpjpe_vals = []
        pmpjpe_vals= []
        for si, (file, n, c) in enumerate(to_run):
            file_path = osp.join(eval_folder, "%s%s.csv" % (file,
                "_file_%s" % m if m != 'all' else ''
            ))
            if osp.exists(file_path):
                print(file_path)
                df = pd.read_csv(file_path)
                a_mpjpe = float(df.loc[df['action'] == 'average']['mpjpe'])
                a_pmpjpe= float(df.loc[df['action'] == 'average']['p-mpjpe'])
                mpjpe_vals.append(a_mpjpe)
                pmpjpe_vals.append(a_pmpjpe)
                distr_names.append(n)
                colors.append(c)
            else:
                print("File not found %s" % file_path)
        
        if ax is not None:
            if mode == 'line':
                lstyle = metrics_styles[mi] if linestyle is None else linestyle
                ax.plot(mpjpe_vals,marker=marker, label=name if m=='all' else "[$MPJPE_{\leq%s}$] %s" % (m, name),
                    linestyle = lstyle, color=color
                )
            elif mode == 'bar':
                for i in range(len(distr_names)):
                    ax.bar(i + shift, mpjpe_vals[i], width = 0.2, color = colors[i], hatch=hatch)
    return mpjpe_vals, pmpjpe_vals

def plot_exp_series(args, exp_series, x_ticks, xlabel="Distortion Type", ylabel="MPJPE (mm)", 
        out_file="mpjpe_plot.png", title="Plot", mode = 'line', metrics = 'all', plot_bounds = None, df_only=False):
    patterns= ['//','+', 'o', '-', '\\', '.', '*', "\\", '//','+', 'o', '-', '\\', '.', '*', "\\"]
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    records    = [] 
    baseline_mpjpe = None
    if df_only == False:
        plt.figure(figsize=(10,6))
    for i, exp_id in enumerate(exp_series):
        folder = exp_series[exp_id]['folder'] if 'folder' in exp_series[exp_id] else exp_id
        color = colors[i % len(colors)]
        lstyle= None
        if plot_bounds is not None:    
            if  i == plot_bounds[0]:
                lstyle = '-.' 
                color  = 'k'
            if len(plot_bounds)==2:
                if i == plot_bounds[1]:
                    lstyle = '-' 
                    color  = 'k'
        
        mpjpe_vals, pmpjpe_vals = plot_exp(plt.gca() if df_only == False else None, 
            osp.join(args.eval_results, folder),
            exp_series[exp_id]['exps'],
            name=exp_series[exp_id]['label'],
            shift= (i - len(exp_series)//2)*0.4 + 0.2,
            hatch= patterns[i],
            mode = mode,
            metrics = metrics,
            color=color,
            linestyle=lstyle,
            marker=markers[i % len(markers)]
        )
        info_dict = {'name': exp_id}
        info_dict.update({k:v for k,v in zip(x_ticks, mpjpe_vals)})
        info_dict['avg_mpjpe'] = round(np.array(mpjpe_vals[1:]).mean(), 2) #skip the original (clean)
        records.append(info_dict)
        if i == 0:
            baseline_mpjpe = mpjpe_vals
        # Compare with baseline model
        info_dict = {'name': '[BASELINE_DIFF] %s' %exp_id}
        mpjpe_vals_diff = [round(x,2) for x in np.array(mpjpe_vals) - baseline_mpjpe]
        info_dict.update({k:v for k,v in zip(x_ticks, mpjpe_vals_diff)})
        info_dict['avg_mpjpe'] = round(np.array(mpjpe_vals_diff[1:]).mean(), 2) #skip the original (clean)
        records.append(info_dict)
    if df_only == False:
        plt.xticks(np.arange(len(x_ticks)), x_ticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if mode == 'bar':
            plt.legend(handles=[mpatches.Patch(hatch=patterns[i], edgecolor='black', facecolor='white',  zorder = 1, label=exp_id) for i, exp_id in enumerate(exp_series)], loc='best')
        else:
            plt.legend(loc='best')
        plt.title(title)
        plt.grid()
        # Export plot
        plt.savefig(osp.join(args.out_folder, out_file), bbox_inches="tight")
        tikzplotlib.save(osp.join(args.out_folder, "%s.tex5" % out_file.split('.')[0]))
    # Export csv
    df = pd.DataFrame.from_records(records, index='name')
    df.to_csv(osp.join(args.out_folder, "%s.csv" % out_file.split('.')[0]))
    return df

def plot_exp_gauss(args, exps_list, out_file="mpjpe_plot.png", 
                    distortion_parts="legs", distortion_temporal = "0.3",  plot_bounds = None):
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
        out_file=out_file,
        plot_bounds = plot_bounds
        )

def plot_time_distortion(args, exps_list, out_file="mpjpe_plot.png", 
                    distortion_parts="legs", distortion_type = "gauss_0.1", plot_bounds=None):
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
        plot_bounds=plot_bounds
        )

def plot_img_distortion(args, exps_list, out_file="mpjpe_img_distortion.png", 
                dist_thr = None, plot_bounds = False):
    exp_series = {}
    for i, (exp, ckpt, label) in enumerate(exps_list):
        model_name = "VideoPose3D"
        color = 'bisque'
        distortion_types  = [
            ('clean', 'gray'),
            ('gaussian_noise',color),
            ('impulse_noise',color),
            ('temporal' ,color),
            ('erase' ,color),
            ('crop' ,color),
            ('motion_blur' ,color),
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
        plot_bounds = plot_bounds
        )

def plot_mpjpe_rel_mpjpe(args, exps_list, out_file="rel-mpjpe-compare.png", metrics =['0.1'], plot_bounds=None,
    detector="hrnet", df_only=False):
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
                '%s_%s_%s_None_None' % (
                    ckpt, detector, t, 
                ), t, c
            )
            )
        exp_series[exp] = {'label': label, 'exps': list_plot_all}
    x_ticks= [i[0] for i in distortion_types]
    title  = "[%s] Performance on distored images input" % (model_name)
    df = plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        metrics = metrics,
        plot_bounds=plot_bounds,
        df_only=df_only
        )
    return df

def plot_mpjpe_humaneva(args, exps_list, out_file="rel-mpjpe-compare.png", metrics =['0.1'], plot_bounds=None,
    detector="hrnet"):
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
                '%s_%s_%s_None_None' % (
                    ckpt, detector, t, 
                ), t, c
            )
            )
        exp_series[exp] = {'label': label, 'exps': list_plot_all}
    x_ticks= [i[0] for i in distortion_types]
    title  = "[%s] Performance on distored images input" % (model_name)
    df = plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        metrics = metrics,
        plot_bounds=plot_bounds
        )
    return df

def plot_mpjpe_lite_hrnet_mpjpe(args, exps_list, out_file="rel-mpjpe-compare.png", metrics =['0.1'], plot_bounds=None, model_name = "VideoPose3D" ):
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
        list_plot_all = []
        for t, c in distortion_types:
            list_plot_all.append((
                '%s_hrnet_%s_None_None' % (
                    ckpt, t, 
                ), t,  c
            )
            )
        exp_series['hrnet_'+exp] = {
            'label': "[TESTED ON HRNET-KPTS] "  + label, 
            'exps': list_plot_all,
            'folder': exp}
        
        list_plot_all = []
        for t, c in distortion_types:
            list_plot_all.append((
                '%s_litehrnet_%s_None_None' % (
                    ckpt, t, 
                ), t,  c
            )
            )
        
        exp_series['litehrnet_'+exp] = {
            'label': "[TESTED ON Lite-HRNET-KPTS] "  + label, 
            'exps': list_plot_all,
            'folder': exp}
        
    x_ticks= [i[0] for i in distortion_types]
    title  = "[%s] Performance on distored images input" % (model_name)
    df = plot_exp_series(args,
        exp_series,
        x_ticks,
        title=title,
        out_file=out_file,
        metrics = metrics,
        plot_bounds=plot_bounds
        )
    return df

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
        ]
    plot_exp_gauss(args, exps_list,
        out_file = "mpjpe_dist_kpts_noise_level.png", plot_bounds=[0]
    )
    plot_time_distortion(args, exps_list,
        out_file = "mpjpe_dist_kpts_temp_distortion.png", plot_bounds=[0]
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
    # exps_list = [
    #     ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
    #     "epoch_80",
    #     "VP3D on CLEAN hrnet det"),
        
    #     ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
    #     "epoch_80",
    #     "VP3D on MIX-AUG hrnet det"),
    # ]
    # plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img.png", metrics=[0.1], plot_bounds=[0,1])

    exps_list = [
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D/H36M "),
        
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.05-dp_rand_0.2-df0.2-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D/H36M, $\sigma=0.05, p,k=20\%$"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D/H36M, $\sigma=0.1, p,k=30\%$"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D/H36M, $\sigma=0.3, p,k=50\%$"),

        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D/dist-H36M-imgs"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img_gauss.png", metrics=[0.1] , plot_bounds=[0,len(exps_list)-1])

def plot_mpjpe_at_0_1_params_analysis(args):
    f, axs = plt.subplots(1, 2, figsize=(10,5))
    for i, k in enumerate([0.1, 0.3, 0.5, 0.8, 1.0]):
        exps_list_parts = []
        exps_list_frames = []
        for p in [0.1, 0.3, 0.5, 0.8, 1.0]:
            exps_list_parts.append(
                ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_%s-df%s-lss_exc_None-conf_None" % (p, k),
                "epoch_80",
                ""),
            )
            exps_list_frames.append(
                ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_%s-df%s-lss_exc_None-conf_None" % (k, p),
                "epoch_80",
                ""),
            )
        df_parts  = plot_mpjpe_rel_mpjpe(args, exps_list_parts, out_file="mpjpe_dist_img_gauss_parts_ratio.png", metrics=[0.1] , plot_bounds=[0,1], df_only=True)
        df_frames = plot_mpjpe_rel_mpjpe(args, exps_list_frames, out_file="mpjpe_dist_img_gauss_frames_ratio.png", metrics=[0.1] , plot_bounds=[0,1], df_only=True)
        axs[0].plot(df_parts['avg_mpjpe'].tolist()[::2], marker=markers[i], label="k=%d" % int(k*100) + '%')
        axs[1].plot(df_frames['avg_mpjpe'].tolist()[::2],marker=markers[i], label="p=%d" % int(k*100) + '%')
    
    tick_labels = [r'$10\%$', r'$30\%$', r'$50\%$', r'$80\%$', r'$100\%$']
    for i in range(2):
        axs[i].set_xticks(np.arange(len(tick_labels)))
        axs[i].set_xticklabels(tick_labels, fontsize=12)
        axs[i].grid()
        axs[i].legend(loc='upper left')
        # axs[i].set_ylim(85,125)
    axs[0].set_ylabel(r"Average MPJPE$_{\leq 0.1}$")
    axs[0].set_xlabel(r"Joints Distortion Ratio p (%)")
    axs[1].set_xlabel(r"Temporal Distortion Ratio k (%)")
    plt.savefig(osp.join(args.out_folder, "ratio_temp_gauss_params.png"), bbox_inches="tight")
    # plt.savefig(osp.join(args.out_folder, "ratio_temp_gauss_params.pdf"), bbox_inches="tight")
    tikzplotlib.save(osp.join(args.out_folder, "ratio_temp_gauss_params.tex"))

def plot_conf_scr_learning(args):
    exps_list = [
        # ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        # "epoch_80",
        # "VP3D on CLEAN hrnet det"),

        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on MIX-AUG hrnet det"),

        # ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        # "epoch_80",
        # "VP3D on CLEAN + CONF SCORE hrnet det"),
        
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "VP3D on MIX-AUG + CONF SCORE hrnet det"),

        # ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det_smthconf",
        # "epoch_80",
        # "VP3D-V0 on MIX-AUG + CONF SCORE NORMALIZED hrnet det"),

        ("ConfVideoPose3DV34gamma0-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 on MIX-AUG gamma=0 hrnet det"),

        ("ConfVideoPose3DV34-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 on MIX-AUG gamma=1 hrnet det"),

        # ("ConfVideoPose3DV34gamma3-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        # "epoch_80",
        # "Conf-VP3D-V34 on MIX-AUG gamma=3 hrnet det"),

        # ("ConfVideoPose3DV34gamma5-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        # "epoch_80",
        # "Conf-VP3D-V34 on MIX-AUG gamma=5 hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img_conf_scr.png", metrics=[0.1], plot_bounds=[0])

def plot_receptive_field(args):
    exps_list = [
         # ----------- 1 FRAME -------------------
        ("VideoPose3D-hrnet_mix-a1,1,1-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (1 frame) on MIX-AUG hrnet det"),

        ("ConfVideoPose3DV34-hrnet_mix-a1,1,1-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 (1 frame) on MIX-AUG gamma=1 hrnet det"),

        # ----------- 9 FRAMES -------------------
        ("VideoPose3D-hrnet_mix-a3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (9 frames) on MIX-AUG hrnet det"),

        ("ConfVideoPose3DV34-hrnet_mix-a3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 (9 frames) on MIX-AUG gamma=1 hrnet det"),

        # ----------- 27 FRAMES -------------------
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (27 frames) on MIX-AUG hrnet det"),

        ("ConfVideoPose3DV34-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 (27 frames) on MIX-AUG gamma=1 hrnet det"),
        
        # ----------- 81 FRAMES -------------------
        ("VideoPose3D-hrnet_mix-a3,3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None", #TRAINING!!!
        "epoch_80",
        "VP3D (81 frames) on dist-H36M-imgs"),

        ("ConfVideoPose3DV34-hrnet_mix-a3,3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 (81 frames) on MIX-AUG gamma=1 hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img_receptive_field_conf.png", metrics=[0.1], plot_bounds=[0])


    exps_list = [
        # ----------- 1 FRAME -------------------
        ("VideoPose3D-hrnet_clean-a1,1,1-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (1 frame) on H36M "),
        
        ("VideoPose3D-hrnet_mix-a1,1,1-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (1 frame) on dist-H36M-imgs"),

        ("VideoPose3D-hrnet_clean-a1,1,1-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (9 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"),

        # ----------- 9 FRAMES -------------------
        ("VideoPose3D-hrnet_clean-a3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (9 frames) on H36M "),
        
        ("VideoPose3D-hrnet_mix-a3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (9 frames) on dist-H36M-imgs"),

        ("VideoPose3D-hrnet_clean-a3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (9 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"),

        # ----------- 27 FRAMES -------------------
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (27 frames) on H36M "),
        
        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (27 frames) on dist-H36M-imgs"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"),

        # ----------- 81 FRAMES -------------------
        ("VideoPose3D-hrnet_clean-a3,3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (81 frames) on H36M"),

        ("VideoPose3D-hrnet_mix-a3,3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (81 frames) on dist-H36M-imgs"),

        ("VideoPose3D-hrnet_clean-a3,3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (81 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"),
    ]
    df_parts = plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_dist_img_receptive_field_gauss.png", metrics=[0.1] , plot_bounds=[0,1])

def plot_lite_hrnet(args):
    exps_list = [
        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D trained on CLEAN hrnet det"),

        ("VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN + noise $\sigma=0.3$ hrnet det."),

        ("VideoPose3D-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (27 frames) on dist-H36M-imgs"),

        ("ConfVideoPose3DV34-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 on MIX-AUG gamma=1 hrnet det"),
    ]
    plot_mpjpe_lite_hrnet_mpjpe(args, exps_list, out_file="mpjpe_trained_on_hrnet.png", metrics=[0.1], plot_bounds=[0])

    exps_list = [
        ("VideoPose3D-litehrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D trained on CLEAN hrnet det"),

        ("VideoPose3D-litehrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D on CLEAN + noise $\sigma=0.3$ hrnet det."),

        ("VideoPose3D-litehrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "VP3D (27 frames) on dist-H36M-imgs"),

        ("ConfVideoPose3DV34-litehrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_det",
        "epoch_80",
        "Conf-VP3D-V34 on MIX-AUG gamma=1 hrnet det"),
    ]
    plot_mpjpe_lite_hrnet_mpjpe(args, exps_list, out_file="mpjpe_trained_on_litehrnet.png", metrics=[0.1], plot_bounds=[0])


def plot_poseformer(args):
    exps_list = [
        ("PoseFormer_27-hrnet_clean-a3,3,3-b10240-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "PoseFormer (27 frames) on CLEAN hrnet det"),

        ("PoseFormer_27-hrnet_clean-a3,3,3-b10240-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "PoseFormer (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"),   

         ("PoseFormer_27-hrnet_mix-a3,3,3-b10240-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "PoseFormer (27 frames) on MIX-AUG hrnet det"),   
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_poseformer.png", metrics=[0.1], plot_bounds=[0])


def plot_attention3dhp(args):
    exps_list = [
        ("Attention3DHP-hrnet_clean-a3,3,3,3,3-b4096-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "Attention3DHP (243 frames) on CLEAN hrnet det"),

        ("Attention3DHP-hrnet_clean-a3,3,3,3,3-b4096-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "Attention3DHP (243 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"), 

        ("Attention3DHP-hrnet_mix-a3,3,3,3,3-b4096-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "Attention3DHP (243 frames) on MIX-AUG hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_attention3dhp.png", metrics=[0.1], plot_bounds=[0])

def plot_srnet(args):
    exps_list = [
        ("SRNet-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "SRNet (27 frames) on CLEAN hrnet det"),

        ("SRNet-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        "epoch_80",
        "SRNet (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"), 

        ("SRNet-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_80",
        "SRNet (27 frames) on MIX-AUG hrnet det"),
    ]
    plot_mpjpe_rel_mpjpe(args, exps_list, out_file="mpjpe_srnet.png", metrics=[0.1], plot_bounds=[0])


def plot_humaneva(args):
    exps_list = [
        ("VideoPose3D-humaneva_hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        "epoch_1000",
        "SRNet (27 frames) on CLEAN hrnet det"),
        
        ("VideoPose3D-humaneva_hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.1-df0.1-lss_exc_None-conf_None",
        "epoch_1000",
        "SRNet (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"), 
        
        ("VideoPose3D-humaneva_hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.3-df0.05-lss_exc_None-conf_None",
        "epoch_1000",
        "SRNet (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"), 

        # ("VideoPose3D-humaneva_hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None",
        # "epoch_1000",
        # "SRNet (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$"),

        # ("VideoPose3D-humaneva_hrnet_clean-a3,3,3-b1024-dj_gauss_0.5-dp_rand_0.05-df0.1-lss_exc_None-conf_None",
        # "epoch_80",
        # "SRNet (27 frames) on H36M, $\sigma=0.3, p=30\%, k=50\%$")

        # ("SRNet-hrnet_mix-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None",
        # "epoch_80",
        # "SRNet (27 frames) on MIX-AUG hrnet det"),
    ]
    plot_mpjpe_humaneva(args, exps_list, out_file="mpjpe_humaneva.png", metrics=['0.1'], plot_bounds=[0], detector="humaneva_hrnet")

def main(args):
    # plot_dist_kpts(args)    
    # plot_dist_imgs(args)
    plot_mpjpe_at_t_img_distortion(args)
    # plot_conf_scr_learning(args)
    # plot_mpjpe_at_0_1_params_analysis(args)
    # plot_receptive_field(args)
    # plot_lite_hrnet(args)
    # plot_poseformer(args)
    # plot_attention3dhp(args)
    # plot_srnet(args)
    # plot_humaneva(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)