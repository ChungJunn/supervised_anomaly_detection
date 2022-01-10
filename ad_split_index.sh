#!/bin/bash

DATASET=$1
DATA_DIR=$HOME"/supervised_anomaly_detection/data/"
RNN_LEN=16
TRAIN_RATIO=0.65 # 0.8
VALID_RATIO=0.10 # 0.2
TEST_RATIO=0.25 # 0.0

python3 ad_split_index.py \
    --dataset=$DATASET \
    --data_dir=$DATA_DIR \
    --rnn_len=$RNN_LEN \
    --train_ratio=$TRAIN_RATIO \
    --valid_ratio=$VALID_RATIO \
    --test_ratio=$TEST_RATIO \