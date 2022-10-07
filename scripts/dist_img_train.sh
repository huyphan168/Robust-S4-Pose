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
    --smooth-conf-score  \
    --no-eval \
    -ste ''