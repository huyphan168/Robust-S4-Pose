#!/bin/bash
python run_exp.py \
    -d humaneva \
    -k hrnet_$1 \
    -arc $2 \
    --drop-conf-score \
    --no-eval \
    -ste '' \