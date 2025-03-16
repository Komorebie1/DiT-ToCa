#!/bin/bash

base_command="python sample.py \
    --image-size 256 \
    --num-sampling-steps 50 \
    --cache-type attention \
    --fresh-threshold 4 \
    --fresh-ratio 0.07 \
    --ratio-scheduler ToCa-ddim50 \
    --force-fresh global \
    --soft-fresh-weight 0.25 \
    --ddim-sample \
    --cluster-nums 16 \
    --cluster-method kmeans \
    --smooth-rate 0.007 \
    --topk 1 \
    "

eval $base_command