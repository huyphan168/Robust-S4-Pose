#!/bin/bash
# python run_exp.py \
#     -str '' \
#     -k hrnet_$1 \
#     -c VideoPose3D-detectron_coco_h36m \
#     -arc 3,3,3,3,3 \
#     --evaluate pretrained_h36m_detectron_coco.bin \
#     --drop-conf-score

python run_exp.py \
    -str '' \
    -k hrnet_$2 \
    -c checkpoint/$1 \
    -arc 3,3,3 \
    --evaluate epoch_80.bin \
    # --drop-conf-score
