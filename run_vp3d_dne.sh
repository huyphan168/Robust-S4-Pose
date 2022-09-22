#!/bin/bash
# cd inference
# python infer_video_d2.py \
#     --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
#     --output-dir output_detection \
#     --image-ext mp4 \
#     input_rgb_videos

# cd data
# python prepare_data_2d_custom.py -i ../inference/output_detection -o myvideos

vid_name=news
python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin \
    --render --viz-subject $vid_name.mp4 \
    --viz-action custom --viz-camera 0 --viz-size 6 \
    --viz-video inference/input_rgb_videos/$vid_name.mp4 \
    --viz-output inference/output_viz/$vid_name.mp4
