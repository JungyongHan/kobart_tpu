#!/bin/bash

PJRT_DEVICE=TPU | ~/venv/bin/python train.py --gradient_clip_val 1.0 \
                --max_epochs 100 \
                --checkpoint checkpoint \
                --accelerator tpu \
                --batch_size 32 \
                --num_workers 4 \
                --precision bf16-true \
                --train_file "./data/train.csv" \
                --test_file "./data/test.csv" 