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
    --cluster-steps 10 \
    --cluster-nums 256 \
    --cluster-method kmeans
    "

eval $base_command