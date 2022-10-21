#!/bin/bash
python run_exp.py \
    -k litehrnet_clean \
    -arc $1 \
    -cfg configs/$2.yaml \
    --drop-conf-score \
    --gpu 0