#!/bin/bash
# if [ -z "$2" ]
#   then
#     CFG=None
#   else
#     CFG=configs/$2.yaml
# fi

# python run_exp.py \
#     -k hrnet_clean \
#     -arc 3,3,3,3 \
#     -cfg $CFG \
#     --no-eval \
#     -ste '' \
#     -m ConfVideoPose3DV34 \

python run_exp.py \
    -k hrnet_clean \
    -arc 3,3,3,3 \
    --drop-conf-score