#!/bin/bash
DISTR_TEMP=0.3
scripts/dist_eval.sh $1 gauss_0.01 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.03 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.05 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.08 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.1 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.2 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.3 $DISTR_TEMP
scripts/dist_eval.sh $1 gauss_0.4 $DISTR_TEMP
