from __future__ import absolute_import, division, print_function

import logging
import torch

from tokenizer import Tokenizer
import torch.nn as nn
from utils import get_start_idxs_batched
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaForMaskedLM, RobertaTokenizer, RobertaModel

logger = logging.getLogger(__name__)

class codebert(nn.Module):
    def __init__(self, model_path, device):
        super(codebert, self).__init__()
        self.tokenizer = Tokenizer.from_pretrained(model_path)
        self.tokenizer.__class__ = Tokenizer         # not perfect convert
        self.model = RobertaModel.from_pretrained(model_path)
        self.hidden_size = 768
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W = nn.Linear(self.hidden_size, 1, bias=False)
        self.block_size = 512
        self.device = device

    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def preprocess(self, inputs, labels, idxs):
        batch_size = len(inputs)
        code = [" ".join(data) for data in inputs]
        code_tokens = self.tokenize(code)
        for i in range(len(code_tokens)):
            code_tokens[i] = [self.tokenizer.cls_token] + code_tokens[i] + [self.tokenizer.sep_token]
        b_start_idxs, b_end_idxs = get_start_idxs_batched(inputs, code_tokens)
        new_labels = []
        for i in range(batch_size):
            if labels[i] == 0:
                idx = 0
            else:
                idx = idxs[i]
            if idx >= len(b_start_idxs[i]):
                new_labels.append((0, 0))
            else:
                new_labels.append((b_start_idxs[i][idx], b_end_idxs[i][idx]))
        block_size = min(512, max(map(len, code_tokens)))
        code_ids = [self.tokenizer.convert_tokens_to_ids(code_tokens[i][:block_size]) for i in range(batch_size)]
        for i in range(batch_size):
            code_ids[i] += [self.tokenizer.pad_token_id] * (block_size - len(code_ids[i]))
        return torch.LongTensor(code_ids).to(self.device), new_labels

    def tokenize(self, inputs, cut_and_pad=False, ret_id=False):
        rets = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for sent in inputs:
            if isinstance(sent, list):
                sent = " ".join(sent)
            if cut_and_pad:
                tokens = self.tokenizer.tokenize(sent)[:self.block_size-2]
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                padding_length = self.block_size - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_length
            else:
                tokens = self.tokenizer.tokenize(sent)
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets

    def _run_batch(self, batch, mask=None):
        batch_max_length = batch.ne(self.tokenizer.pad_token_id).sum(-1).max().item()
        inputs = batch[:, :batch_max_length]
        inputs = inputs.to(self.device)
        attn_mask = inputs.ne(self.tokenizer.pad_token_id).to(inputs)
        lens = torch.sum(attn_mask, -1)
        outputs = self.model(inputs, attention_mask=attn_mask)
        hiddens = outputs[0]    # (B, L, H)
        if mask == None:
            mask = attn_mask    # (B, L, 1)
        mask = mask.unsqueeze(-1)
        mask = mask.to(hiddens)
        hiddens = hiddens * mask
        Hn = self.batched_index_select(hiddens, 1, (lens - 1).unsqueeze(-1))
        Hn = Hn * mask
        mask = mask.to(dtype=bool)
        M = torch.nn.Tanh()(self.W1(hiddens) + self.W2(Hn))
        M = self.W(M).masked_fill(~mask, -1e10).squeeze(-1)    # (B, L)
        return torch.nn.LogSoftmax(dim=1)(M)
    
    def _loss(self, outputs, labels):
        return -torch.sum(outputs * labels)
