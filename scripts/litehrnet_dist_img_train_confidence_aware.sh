#!/bin/bash
python run_exp.py \
    -k litehrnet_mix \
    -arc $2 \
    --no-eval \
    -ste '' \
    -m $1 \
    # --gpu 1