#!/bin/bash
python run_exp.py \
    -k hrnet_clean \
    -arc 3,3,3 \
    -cfg configs/$1.yaml \
    --drop-conf-score \
    -m SRNet \
    -b 1024 \
    --no-eval \
    -ste '' \