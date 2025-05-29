#!/bin/bash

PJRT_DEVICE=TPU | ~/venv/bin/python train.py --gradient_clip_val 1.0 \
                --max_epochs 100 \
                --checkpoint checkpoint \
                --accelerator tpu \
                --num_cores 32 \
                --batch_size 32 \
                --num_workers 4 \
                --precision bf16-mixed \
                --train_file "../summary/dataset/news_data.csv" \
                --test_file "../summary/dataset/2022_2025.csv" 