#!/bin/bash
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 clean
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 erase
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 temporal
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 crop
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 fog
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 brightness
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 gaussian_noise
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 motion_blur
scripts/srnet_dist_img_eval_temporal_gauss_noise.sh $1 impulse_noise