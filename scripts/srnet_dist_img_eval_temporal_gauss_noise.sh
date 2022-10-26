#!/bin/bash
python run_exp.py \
    -k hrnet_$2 \
    -c checkpoint/$1 \
    -arc 3,3,3 \
    -m SRNet \
    --evaluate epoch_80.bin \
    --drop-conf-score \
    --eval-ignore-parts file_0.1 \
    -str '' \
    # --test-fixed-size-input -no-tta\