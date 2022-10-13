#!/bin/bash
if [ -z "$2" ]
  then
    CFG=None
  else
    CFG=configs/$2.yaml
fi

python run_exp.py \
    -k hrnet_$1 \
    -arc 3,3,3 \
    -cfg $CFG \
    --no-eval \
    -ste '' \
    -c checkpoint/draft \
    -m ConfVideoPose3DV34 \
    # --smooth-conf-score  \