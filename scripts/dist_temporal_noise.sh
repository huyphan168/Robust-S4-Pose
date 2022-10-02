#!/bin/bash
SGM=0.1
scripts/dist_eval.sh $1 gauss_$SGM 1.0
scripts/dist_eval.sh $1 gauss_$SGM 0.9
scripts/dist_eval.sh $1 gauss_$SGM 0.8
scripts/dist_eval.sh $1 gauss_$SGM 0.6
scripts/dist_eval.sh $1 gauss_$SGM 0.5
scripts/dist_eval.sh $1 gauss_$SGM 0.5
scripts/dist_eval.sh $1 gauss_$SGM 0.4
scripts/dist_eval.sh $1 gauss_$SGM 0.3
scripts/dist_eval.sh $1 gauss_$SGM 0.2
scripts/dist_eval.sh $1 gauss_$SGM 0.1
scripts/dist_eval.sh $1 gauss_$SGM 0.05
scripts/dist_eval.sh $1 gauss_$SGM 0.01
