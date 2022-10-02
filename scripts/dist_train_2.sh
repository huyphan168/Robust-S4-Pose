#!/bin/bash
python run_exp.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3 \
    -cfg configs/gauss-0.05_parts-rand-0.2_temp-0.2-conf-none.yaml \
    --gpu 0