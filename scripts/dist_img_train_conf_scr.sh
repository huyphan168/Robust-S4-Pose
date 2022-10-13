#!/bin/bash
python run_exp.py \
    -k hrnet_$1 \
    -arc 3,3,3 \
    --no-eval \
    -ste '' \
    -m $2 \