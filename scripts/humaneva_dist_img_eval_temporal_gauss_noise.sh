python run_exp.py \
    -d humaneva \
    -k humaneva_hrnet_$2 \
    -c checkpoint/$1 \
    -arc 3,3,3 \
    --evaluate epoch_80.bin \
    --drop-conf-score \
    -str ''\
    -ste Validate/S1,Validate/S2,Validate/S3 \
    -a Walk,Jog,Box \
    --eval-ignore-parts file_0.1 \
    # --by-subject