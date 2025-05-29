PJRT_DEVICE=TPU | ~/venv/bin/python train.py --gradient_clip_val 1.0 \
        --max_epochs 100 \
        --checkpoint checkpoint \
        --accelerator tpu \
        --num_gpus 16 \
        --batch_size 8 \
        --num_workers 4 \
		--train_file "./data/train.csv" \
		--test_file "./data/test.csv"
