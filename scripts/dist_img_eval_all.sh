#!/bin/bash
scripts/dist_img_eval.sh $1 clean
scripts/dist_img_eval.sh $1 erase
scripts/dist_img_eval.sh $1 temporal
scripts/dist_img_eval.sh $1 crop
scripts/dist_img_eval.sh $1 fog
scripts/dist_img_eval.sh $1 brightness
scripts/dist_img_eval.sh $1 gaussian_noise
scripts/dist_img_eval.sh $1 motion_blur
scripts/dist_img_eval.sh $1 impulse_noise