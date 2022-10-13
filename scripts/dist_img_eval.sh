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
    --eval-ignore-parts file_0.1 \
    -m ConfVideoPose3DV36
    # --smooth-conf-score -no-tta\
    # --drop-conf-score \
    