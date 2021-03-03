#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   BILSTM.py
@Time    :   2021/03/03 17:12:37
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch 


class BiLSTM(object):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, dropout=0.1):
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.bilstm = torch.nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(2 * hidden_size, out_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        emb = torch.nn.utils.rnn.pad_packed_sequence(emb, lengths, batch_first=True)
        emb, _ = self.bilstm(emb)
        emb, _ = torch.nn.utils.rnn.pad_packed_sequence(emb, batch_first=True,padding_value=0.,total_length=x.shape[1])
        scores = self.fc(emb)
        return scores

    def test(self, x, lengths):
        logits = self.forward(x, lengths)
        # 输出每行的最大值及最大值索引
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids


def cal_loss(logits, targets, tag2id):
    

