python run_exp.py \
    -k hrnet_$2 \
    -c checkpoint/$1 \
    -arc 3,3,3 \
    --evaluate epoch_80.bin \
    --drop-conf-score \
    --eval-ignore-parts file_0.1 \
    -m PoseFormer_27 \
    -no-tta \
    --subjects-test S1,S5
    # -str '' \