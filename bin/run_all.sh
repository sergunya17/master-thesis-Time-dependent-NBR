#!/usr/bin/env bash

for DATASET in dunnhumby instacart tafeng
do
    for MODEL in g_top_freq p_top_freq gp_top_freq user_knn als up_cf up_cf_time up_cf_time_next_ts tifuknn tifuknn_time_days tifuknn_time_days_next_ts
    do
        OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python src/scripts/experiment.py \
            --dataset=$DATASET \
            --model=$MODEL \
            --num-trials=400 \
            --batch-size=20000 &
    done
done
