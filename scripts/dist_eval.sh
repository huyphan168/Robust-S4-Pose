#!/bin/bash
if [ -z "$3" ]
  then
    DISTR_TEMP=None
  else
    DISTR_TEMP=$3
fi
# python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3 -c checkpoint/VideoPose3D-cpn_ft_h36m_dbb-a3,3,3-b1024-dj_gauss_0.1-dp_rand_0.3-df0.3-lss_exc_None-conf_gauss --evaluate epoch_80.bin \
python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3 -c checkpoint/$1 \
    --evaluate epoch_80.bin \
    --test-distortion-type $2 \
    --test-distortion-parts legs \
    --eval-ignore-parts legs\
    --test-distortion-temporal $DISTR_TEMP \
    --test-gen-conf-score gauss \