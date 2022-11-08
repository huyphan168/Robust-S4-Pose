#!/bin/bash
python run_exp.py \
    -str '' \
    -k hrnet_clean \
    -c checkpoint/s4_baseline_single/ \
    -arc 3,3,3,3 \
    --evaluate epoch_65.bin \
    --test-fixed-size-input \
    --drop-conf-score \
    --cfg configs/s4_baseline_single.yaml -m S4