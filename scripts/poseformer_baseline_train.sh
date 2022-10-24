python run_exp.py \
    -k hrnet_$2 \
    -m PoseFormer_$1 \
    --drop-conf-score \
    --parallel \
    -b 4096 --gpu 0,1,2,3 \
    --checkpoint-frequency 20\
    --no-eval -ste ''\