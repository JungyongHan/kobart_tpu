import argparse
import numpy as np
import pandas as pd
import os
from loguru import logger

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Import TPU-specific modules
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from dataset import KobartSummaryModule
from model import KoBARTConditionalGeneration

from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description='KoBART Summarization for TPU')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/test.tsv',
                            help='train file')
        parser.add_argument('--test_file',
                            type=str,
                            default='data/test.tsv',
                            help='test file')
        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='batch size per TPU core')
        parser.add_argument('--checkpoint',
                            type=str,
                            default='checkpoint',
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        parser.add_argument('--max_epochs',
                            type=int,
                            default=10,
                            help='train epochs')
        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')
        parser.add_argument('--accelerator',
                            type=str,
                            default='tpu',
                            choices=['tpu', 'gpu', 'cpu'],
                            help='select accelerator')
        parser.add_argument('--num_cores',
                            type=int,
                            default=8,
                            help='number of TPU cores')
        parser.add_argument('--gradient_clip_val',
                            type=float,
                            default=1.0,
                            help='gradient_clipping')
        parser.add_argument('--resume_from_checkpoint',
                            action='store_true',
                            help='resume training from last checkpoint')
        parser.add_argument('--precision',
                            type=str,
                            default='bf16-mixed',
                            help='precision for training (16, 32, bf16-mixed)')

        return parser

# Main training function to be executed in each TPU process
def train_kobart(rank, args):
    # Set the device to the current TPU core
    device = xm.xla_device()
    
    # Initialize tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    
    # 개행문자를 특수 토큰으로 추가
    special_tokens_dict = {'additional_special_tokens': ['<newline>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    logger.info(f"Process {rank}: Initializing data module")
    dm = KobartSummaryModule(args.train_file,
                        args.test_file,
                        tokenizer,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=4)
    dm.setup("fit")
    
    logger.info(f"Process {rank}: Initializing model")
    model = KoBARTConditionalGeneration(args)
    
    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=args.checkpoint,
                                          filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                          verbose=True,
                                          save_last=True,
                                          mode='min',
                                          save_top_k=3)
    
    # Initialize wandb logger only on the main process
    wandb_logger = None
    if rank == 0:
        wandb_logger = WandbLogger(project="KoBART-summary-TPU")

    # Check for checkpoint to resume training
    ckpt_path = None
    if args.resume_from_checkpoint:
        ckpt_path = f"{args.checkpoint}/last.ckpt"
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        logger.info(f"Process {rank}: Resuming from checkpoint: {ckpt_path}")

    # Configure TPU-specific trainer settings
    trainer = L.Trainer(max_epochs=args.max_epochs,
                        accelerator=args.accelerator,
                        devices=args.num_cores,
                        num_nodes=1,  # Using a single TPU VM with multiple cores
                        precision=args.precision,  # Use bfloat16 for TPU
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[checkpoint_callback],
                        logger=wandb_logger,
                        strategy="xla",  # Use XLA strategy for TPU
                        sync_batchnorm=True,  # Synchronize batch normalization across cores
                        )
    
    logger.info(f"Process {rank}: Starting training")
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    
    # Save the model on the main process
    if rank == 0:
        logger.info("Training completed, saving the final model")

if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
    logger.info(args)
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
        os.makedirs(os.path.join(args.checkpoint, "model_chp"), exist_ok=True)
    
    # Launch training across TPU cores using XLA multiprocessing
    xmp.spawn(train_kobart, args=(args,), nprocs=args.num_cores)