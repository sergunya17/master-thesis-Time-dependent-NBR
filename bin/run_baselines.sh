#!/usr/bin/env bash

for DATASET in dunnhumby instacart tafeng
do
    for MODEL in g_top_freq p_top_freq gp_top_freq user_knn als
    do
        OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python src/scripts/experiment.py \
            --dataset=$DATASET \
            --model=$MODEL \
            --num-trials=400 \
            --batch-size=20000 &
    done
done
