#!/bin/bash
python run_exp.py \
    -d humaneva \
    -k humaneva_hrnet_$1 \
    -arc 3,3,3 \
    --drop-conf-score \
    -str Train/S1,Train/S2,Train/S3 \
    -ste Validate/S1,Validate/S2,Validate/S3 \
    # -a Walk,Jog,Box --by-subject
    # --no-eval -ste '' \