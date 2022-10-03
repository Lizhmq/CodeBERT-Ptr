# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import gc
import shutil
import json

from utils import get_start_idxs_batched
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=512):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_langs_%s"%(args.langs)+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels = datas["inputs"], datas["labels"]

        else:
            self.inputs = []
            self.labels = []
            if file_type != "test":
                datafile = os.path.join(args.data_dir, "train.pkl")
                datafile2 = os.path.join(args.data_dir, "valid.pkl")
            else:
                datafile = os.path.join(args.data_dir, "test.pkl")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            datas = pickle.load(open(datafile, "rb"))
            datas2 = pickle.load(open(datafile2, "rb"))
            labels = datas["label"]
            inputs = datas["norm"]
            idxs = datas["idx"]
            
            # LENS = int(len(datas2["label"]) * 0.1)
            LENS = len(datas2["label"]
            labels2 = datas2["label"][:LENS]
            inputs2 = datas2["norm"][:LENS]
            idxs2 = datas2["idx"][:LENS]

            if file_type == "train":
                inputs, labels, idxs = inputs, labels, idxs
            elif file_type == "dev":
                inputs, labels, idxs = inputs2, labels2, idxs2
            length = len(inputs)

            for idx, (data, label, error) in enumerate(zip(inputs, labels, idxs)):
                if idx % world_size == local_rank:
                    code = " ".join(data)
                    code_tokens = tokenizer.tokenize(code)
                    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
                    b_start_idxs, b_end_idxs = get_start_idxs_batched([data], [code_tokens])
                    b_start_idxs, b_end_idxs = b_start_idxs[0], b_end_idxs[0]
                    if len(b_start_idxs) == 0:
                        continue
                    start, end = 0, 0
                    if label:
                        if error >= len(b_start_idxs):
                            start, end = 0, 1
                        else:
                            start, end = b_start_idxs[error], b_end_idxs[error]
                    code_ids = tokenizer.convert_tokens_to_ids(code_tokens[:block_size])
                    padding_length = block_size - len(code_ids)
                    code_ids += [tokenizer.pad_token_id] * padding_length
                    out_label = [0] * block_size
                    if label == 0:
                        out_label[0] = 1
                    else:
                        out_label[start:end] = [1] * (end - start)
                    self.inputs.append(code_ids)
                    self.labels.append(out_label)

                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))

            if file_type == 'train':
                logger.warning("Rank %d Training %d samples"%(local_rank, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump({"inputs": self.inputs, "labels": self.labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item])
