#!/bin/bash
python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3 \
    -c checkpoint/VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None \
    -c2 checkpoint/VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None \
    --evaluate epoch_80.bin --render \
    --drop-conf-score \
    -k hrnet_erase \
    -str '' \
    --viz-subject S11 \
    --viz-action 'WalkDog' \
    --viz-camera 2 \
    --viz-output output.gif --viz-size 3 --viz-downsample 2  \
    --viz-limit 300
# Greeting

    # -c checkpoint/VideoPose3D-hrnet_clean-a3,3,3-b1024-dj_gauss_0.3-dp_rand_0.5-df0.5-lss_exc_None-conf_None \