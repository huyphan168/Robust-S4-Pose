#!/bin/bash
scripts/dist_eval.sh $1 None 0.3
scripts/dist_gauss_noise.sh $1
scripts/dist_temporal_noise.sh $1