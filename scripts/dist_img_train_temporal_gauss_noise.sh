#!/bin/bash
if [ -z "$2" ]
  then
    ARC=3,3,3
  else
    ARC=$2
fi

python run_exp.py \
    -k hrnet_clean \
    -arc $ARC \
    -cfg configs/$1.yaml \
    --drop-conf-score