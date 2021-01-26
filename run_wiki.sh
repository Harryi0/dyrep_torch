#!/bin/sh

python train_eval_main.py \
    --data_dir ./Jodie \
    --dataset wikipedia \
    --all_comms True \
    --device cuda \
    --lr 0.001