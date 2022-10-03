#!/bin/bash
python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin --render \
    --viz-subject S11 \
    --viz-action Greeting \
    --viz-camera 2 \
    --viz-output output.gif --viz-size 3 --viz-downsample 2 --viz-limit 180 \
    --distortion-type $1 \
    --distortion-parts legs \
    --distortion-temporal $DISTR_TEMP 