#!/bin/bash
python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3 \
    -str '' \
    -c checkpoint/$1 \
    -k hrnet_$2 \
    --evaluate epoch_80.bin\
    --render \
    --viz-subject S9 \
    --viz-action Walking \
    --viz-camera 0 \
    --viz-output output.gif --viz-size 3 --viz-downsample 3 \
    --drop-conf-score
    # --viz-limit 200 \
    