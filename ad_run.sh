#!/bin/bash

# logging hyper-params
EXP_NAME="code_cleaning"
USE_NEPTUNE=0

# dataset
DATASET="wsd" #cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
LABEL='sla'
RNN_LEN=16
if [ $DATASET == 'wsd' ] || [ $DATASET == 'cnsm_exp2_1' ] || [ $DATASET == 'cnsm_exp2_2' ]
then
    DIM_INPUT=24 # added 1 for label information
else
    echo '$DATASET must be either wsd, lad1, or lad2'
    exit -1
fi
BASE_DIR=$HOME'/supervised_anomaly_detection/data/'
CSV_PATH=$BASE_DIR''$DATASET'.csv'
IDS_PATH=$BASE_DIR''$DATASET'.indices.rnn_len16.pkl'
STAT_PATH=$CSV_PATH'.stat'
DATA_NAME=$DATASET # TODO: remove

# encoder hyper-params
DIM_FEATURE_MAPPING=24
ENCODER="rnn"
NLAYER=2
DIM_ENC=-1              # DNN-enc
BIDIRECTIONAL=0         # RNN-enc
DIM_LSTM_HIDDEN=40      # RNN-enc
NHEAD=4                 # transformer
DIM_FEEDFORWARD=48      # transformer
REDUCE="self-attention" # mean, max, or self-attention

# classifier hyper-params
CLASSIFIER="rnn" # dnn or rnn
CLF_N_LSTM_LAYERS=1
CLF_N_FC_LAYERS=3
CLF_DIM_LSTM_HIDDEN=200
CLF_DIM_FC_HIDDEN=600

# modified model hyper-params
TEACHER_FORCING_RATIO=0.5

if [ $LABEL == 'sla' ] # TODO: remove this
then
    CLF_DIM_OUTPUT=2
else
    echo '$LABEL must be either sla'
    exit -1
fi

# optim hyper-params
OPTIMIZER='Adam'
LR=0.001
DROP_P=0.0
BATCH_SIZE=64
PATIENCE=10
MAX_EPOCH=1
USE_SCHEDULER=0
STEP_SIZE=1
GAMMA=0.5
N_DECAY=3

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDBmMTBmOS0zZDJjLTRkM2MtOTA0MC03YmQ5OThlZTc5N2YifQ=="
export CUDA_VISIBLE_DEVICES=$1

/usr/bin/python3.8 ad_main.py \
    --reduce=$REDUCE \
    --optimizer=$OPTIMIZER \
    --lr=$LR \
    --patience=$PATIENCE \
    --exp_name=$EXP_NAME \
    --dataset=$DATASET \
    --max_epoch=$MAX_EPOCH \
    --batch_size=$BATCH_SIZE \
    --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
    --dim_feature_mapping=$DIM_FEATURE_MAPPING \
    --nlayer=$NLAYER \
    --bidirectional=$BIDIRECTIONAL \
    --nhead=$NHEAD \
    --dim_feedforward=$DIM_FEEDFORWARD \
    --dim_input=$DIM_INPUT \
    --encoder=$ENCODER \
    --classifier=$CLASSIFIER \
    --dim_enc=$DIM_ENC \
    --clf_n_lstm_layers=$CLF_N_LSTM_LAYERS \
    --clf_n_fc_layers=$CLF_N_FC_LAYERS \
    --clf_dim_lstm_hidden=$CLF_DIM_LSTM_HIDDEN \
    --clf_dim_fc_hidden=$CLF_DIM_FC_HIDDEN \
    --clf_dim_output=$CLF_DIM_OUTPUT \
    --csv_path=$CSV_PATH \
    --ids_path=$IDS_PATH \
    --stat_path=$STAT_PATH \
    --data_name=$DATA_NAME \
    --rnn_len=$RNN_LEN \
    --label=$LABEL \
    --dict_path=$DICT_PATH \
    --use_neptune=$USE_NEPTUNE \
    --use_scheduler=$USE_SCHEDULER \
    --step_size=$STEP_SIZE \
    --gamma=$GAMMA \
    --n_decay=$N_DECAY \
    --drop_p=$DROP_P \
    --teacher_forcing_ratio=$TEACHER_FORCING_RATIO