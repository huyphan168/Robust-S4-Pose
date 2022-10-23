#!/bin/bash
python run_exp.py \
    -k hrnet_$2 \
    -m PoseFormer_$1 \
    --drop-conf-score \
    --parallel \
    -b 2048 -e 60 --checkpoint-frequency 40 \
    -lr 0.0004 -lrd 0.99 \
    -ste 'S9' \
    # --no-eval \
    