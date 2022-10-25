#!/bin/bash
python run_exp.py \
    -k hrnet_$1 \
    -arc 3,3,3 \
    --drop-conf-score \
    --no-eval \
    -ste '' \
    -m SRNet \
    -b 1024