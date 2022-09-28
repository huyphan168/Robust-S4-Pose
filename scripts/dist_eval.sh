#!/bin/bash
if [ -z "$2" ]
  then
    DISTR_TEMP=None
  else
    DISTR_TEMP=$2
fi
# 

# python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3 -c checkpoint/VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_gauss --evaluate epoch_80.bin \
python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3 -c checkpoint/VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_None-dp_None-dfNone-lss_exc_None-conf_None --evaluate epoch_80.bin \
    --test-distortion-type $1 \
    --test-distortion-parts legs \
    --test-distortion-temporal $2 \
    # --test-gen-conf-score binary \