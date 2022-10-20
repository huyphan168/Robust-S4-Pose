#!/bin/bash
python run_exp.py \
    -k hrnet_$2 \
    -m PoseFormer_$1 \
    --drop-conf-score \
    --parallel