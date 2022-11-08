#!/bin/bash
python run_exp.py \
    -k hrnet_clean \
    --drop-conf-score \
    -cfg configs/$1.yaml \
    -m S4 \
    --no-eval \
    -b 1024 