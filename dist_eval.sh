#!/bin/bash
python run_exp.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin \
    --distortion-type $1 \
    --distortion-parts arm-right