#!/bin/bash
python run_exp.py \
    -str '' \
    -k litehrnet_$2 \
    -c checkpoint/$1 \
    -arc 3,3,3 \
    --evaluate epoch_80.bin \
    --eval-ignore-parts file_0.1 \
    -m ConfVideoPose3DV34