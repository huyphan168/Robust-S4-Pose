#!/bin/bash
# python run_exp.py \
#     -k hrnet_$1 \
#     -arc 3,3,3 \
#     --drop-conf-score 

python run_exp.py \
    -k hrnet_$1 \
    -arc 3,3,3 \
    # -cfg configs/$1.yaml \
    --drop-conf-score 
