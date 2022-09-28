#!/bin/bash
python run_exp.py \
    -d h36m \
    -str '' \
    -ste 'S9' \
    -k hrnet_erase \
    -arc 3,3,3 \
    -c checkpoint/VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None \
    --evaluate epoch_80.bin
