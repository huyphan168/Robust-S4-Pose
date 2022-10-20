#!/bin/bash
scripts/dist_img_eval.sh $1 clean $2
scripts/dist_img_eval.sh $1 erase $2
scripts/dist_img_eval.sh $1 temporal $2
scripts/dist_img_eval.sh $1 crop $2
scripts/dist_img_eval.sh $1 fog $2
scripts/dist_img_eval.sh $1 brightness $2
scripts/dist_img_eval.sh $1 gaussian_noise $2
scripts/dist_img_eval.sh $1 motion_blur $2
scripts/dist_img_eval.sh $1 impulse_noise $2