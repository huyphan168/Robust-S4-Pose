#!/bin/bash
python run_exp.py \
    -k hrnet_clean \
    -arc 3,3,3,3,3 \
    -cfg configs/$1.yaml \
    --drop-conf-score \
    --no-eval \
    -ste '' \
    -m Attention3DHP \
    -b 2048