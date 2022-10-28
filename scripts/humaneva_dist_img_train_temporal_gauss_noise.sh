#!/bin/bash
python run_exp.py \
    -d humaneva \
    -k humaneva_hrnet_clean \
    -arc $1 \
    -cfg configs/$2.yaml \
    --drop-conf-score \
    -str Train/S1,Train/S2,Train/S3 \
    -ste Validate/S1,Validate/S2,Validate/S3 \