#!/bin/bash
python run_exp.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3 \
    -cfg configs/msk_gauss_conf.yaml \
    --gpu 0