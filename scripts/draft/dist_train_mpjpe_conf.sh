#!/bin/bash
python run_exp.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3 \
    -cfg configs/gauss-0.1_parts-rand-0.3_temp-0.3_conf-gauss.yaml \
    -l conf_mpjpe \
    --gpu 0