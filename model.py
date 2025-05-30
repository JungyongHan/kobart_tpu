import torch
import argparse
from collections import defaultdict

import lightning as L
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from loguru import logger

class KoBARTConditionalGeneration(L.LightningModule):
    def __init__(
        self,
        hparams,
        **kwargs):
        
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.newline_token = '<newline>'
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        # 개행문자를 특수 토큰으로 추가
        special_tokens_dict = {'additional_special_tokens': [self.newline_token]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.pad_token_id = self.tokenizer.pad_token_id
        
        self.outputs = defaultdict(list)
            
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr)
        num_workers = self.hparams.num_workers

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    
    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        # Use XLA's mark_step to optimize TPU performance
        xm.mark_step()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outs = self(batch)
        loss = outs['loss']
        # Use XLA's mark_step to optimize TPU performance
        xm.mark_step()
        self.outputs[dataloader_idx].append({"loss": loss})

    def on_validation_epoch_end(self):
        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.outputs.clear()