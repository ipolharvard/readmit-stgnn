#!/bin/bash

python stgnn/train.py \
    --save_dir stgnn/data/ckp \
    --demo_file stgnn/data/mimic_processed/mimic_admission_demo.csv \
    --edge_modality 'demo' \
    --feature_type 'non-imaging' \
    --ehr_feature_file stgnn/data/mimic_processed/ehr_preprocessed_seq_by_day.pkl \
    --edge_ehr_file stgnn/data/mimic_processed/ehr_preprocessed_seq_by_day.pkl \
    --ehr_types 'demo' 'icd' 'lab' 'med' \
    --edge_top_perc 0.1 \
    --max_seq_len_ehr 9 \
    --train_batch_size 128 \
    --hidden_dim 256 \
    --num_rnn_layers 3 \
    --dropout 0.2 \
    --metric_name auroc \
    --lr 0.001 \
    --l2_wd 5e-4 \
    --patience 10 \
    --pos_weight 1 \
    --num_epochs 100 \
    --model_name rnn \
    --ehr_encoder_name 'embedder'
