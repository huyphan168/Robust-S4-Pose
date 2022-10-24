#!/bin/bash
python run_exp.py \
    -k hrnet_clean \
    -m PoseFormer_27 \
    --drop-conf-score \
    --parallel \
    --no-eval \
    -ste '' \
    -b 8192 \
    --checkpoint-frequency 20 \
    --gpu 0,1,2,3\
    --cfg configs/gauss-0.3_parts-rand-0.5_temp-0.5_conf-none.yaml \