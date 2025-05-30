import argparse
import os
import re
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import lightning as L
from functools import partial

# Import TPU-specific modules
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file)
        # 개행문자를 위한 특수 토큰 정의
        self.newline_token = '<newline>'
        self.docs = self.preprocess_data(self.docs)
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    # 데이터 전처리 함수 정의
    def preprocess_data(self, data):
        # NaN 값 제거
        original_len = len(data)
        data['article'] = data['article'].astype(str)
        data['script'] = data['script'].astype(str)

        data['article_len'] = data['article'].apply(len)
        data['script_len'] = data['script'].apply(len)

        data = data[data['article_len'] > data['script_len']]
        filtered_len = len(data)
        print(f"데이터 필터링: {original_len}개 중 {filtered_len}개 남음 ({original_len - filtered_len}개 제외)")
        
        def convert_br_to_blank(text):
            return text.replace('<br>', '')

        # <br> 태그를 실제 개행 문자로 변환
        def convert_br_to_newline(text):
            return text.replace('<br>', '\n')
        
        # HTML 태그 제거
        def clean_html(text):
            return re.sub(r'<[^>]+>', '', text)
        
        # 이메일 주소 제거
        def remove_email(text):
            return re.sub(r'\S+@\S+\.\S+', '', text)
        
        # 웹사이트 주소 제거
        def remove_urls(text):
            return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
        # 괄호 내용 제거 (소괄호, 대괄호, 중괄호)
        def remove_brackets(text):
            text = re.sub(r'\([^)]*\)', '', text)  # 소괄호
            text = re.sub(r'\[[^\]]*\]', '', text)  # 대괄호
            text = re.sub(r'\{[^}]*\}', '', text)  # 중괄호
            return text
        
        # 특수문자 제거 (말줄임표, 중간점 등)
        def remove_special_chars(text):
            # 특수문자 제거 (알파벳, 숫자, 한글, 공백, 개행문자 외 모두 제거)
            text = re.sub(r'[^\w\s가-힣\n]', '', text)
            return text
        
        # 여러 공백을 하나로 치환 (개행문자는 유지)
        def clean_spaces(text):
            text = re.sub(r' +', ' ', text)  # 여러 공백을 하나로
            text = re.sub(r'\n+', '\n', text)  # 여러 개행을 하나로
            return text.strip()

        # 전처리 적용 - <br>을 실제 개행문자로 변환
        data['article'] = data['article'].apply(convert_br_to_blank)
        data['script'] = data['script'].apply(convert_br_to_newline)
        
        # 나머지 전처리 적용
        data['article'] = data['article'].apply(clean_html)
        data['article'] = data['article'].apply(remove_email)
        data['article'] = data['article'].apply(remove_urls)
        data['article'] = data['article'].apply(remove_brackets)
        data['article'] = data['article'].apply(remove_special_chars)
        data['article'] = data['article'].apply(clean_spaces)

        data['script'] = data['script'].apply(clean_html)
        data['script'] = data['script'].apply(clean_spaces)
        
        data = data.drop(['article_len', 'script_len'], axis=1)
        return data

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        # add special tokens \n will be '<newline>'
        instance['script'] = instance['script'].replace('\n', self.newline_token)
        input_ids = self.tokenizer.encode(instance['article'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['script'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)
               }

    def __len__(self):
        return self.len

class KobartSummaryModule(L.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=512,
                 batch_size=8,
                 num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTSummaryDataset(self.train_file_path,
                                 self.tok,
                                 self.max_len)
        self.test = KoBARTSummaryDataset(self.test_file_path,
                                self.tok,
                                self.max_len)

    def train_dataloader(self):
        # Create TPU-optimized DataLoader using ParallelLoader
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        
        train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers
        )
        
        # Wrap with ParallelLoader for TPU optimization
        train_device_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())
        return train_device_loader

    def val_dataloader(self):
        # Create TPU-optimized DataLoader using ParallelLoader
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )
        
        val_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers
        )
        
        # Wrap with ParallelLoader for TPU optimization
        val_device_loader = pl.MpDeviceLoader(val_loader, xm.xla_device())
        return val_device_loader

    def test_dataloader(self):
        # Create TPU-optimized DataLoader using ParallelLoader
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )
        
        test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=self.num_workers
        )
        
        # Wrap with ParallelLoader for TPU optimization
        test_device_loader = pl.MpDeviceLoader(test_loader, xm.xla_device())
        return test_device_loader