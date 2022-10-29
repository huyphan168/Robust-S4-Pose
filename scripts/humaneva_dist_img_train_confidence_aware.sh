#!/bin/bash
python run_exp.py \
    -d humaneva \
    -k humaneva_hrnet_$1 \
    -arc 3,3,3 \
    -lrd 0.996 \
    -e 1000 \
    --checkpoint-frequency 200 \
    -m ConfVideoPose3DV34 \
    -str Train/S1,Train/S2,Train/S3 \
    -ste Validate/S1,Validate/S2,Validate/S3 \
    # --gpu 1