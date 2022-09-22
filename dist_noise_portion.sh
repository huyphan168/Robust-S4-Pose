#!/bin/bash
SGM=0.1
./dist_eval.sh gauss_$SGM 1.0
./dist_eval.sh gauss_$SGM 0.5
./dist_eval.sh gauss_$SGM 0.4
./dist_eval.sh gauss_$SGM 0.3
./dist_eval.sh gauss_$SGM 0.2
./dist_eval.sh gauss_$SGM 0.1
./dist_eval.sh gauss_$SGM 0.05
./dist_eval.sh gauss_$SGM 0.01
