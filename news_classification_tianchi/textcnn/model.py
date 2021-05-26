#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/05/24 16:17:02
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from base import * 
import torch.nn.functional as F 


class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    
    def forward(self, batch_hidden, batch_masks):
        key = torch.matmul(batch_hidden, self.weight) + self.bias
        outputs = torch.matmul(key, self.query)
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))
        attn_scores = F.softmax(masked_outputs, dim=1)

        masked_attn_scores = attn_scores.masked_fill((1-batch_masks).bool(), 0.0)
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)
        return batch_outputs, attn_scores


class WordCNNEncoder(torch.nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.word_dims = word_dims
        self.word_embed = torch.nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        input_size = self.word_dims
        self.filter_sizes = [2,3,4]
        self.out_channel = 100
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True) for filter_size in self.filter_sizes])

    def forward(self, word_ids):
        sen_num, sen_len = word_ids.shape
        word_embed = self.word_embed(word_ids)
        batch_embed = word_embed
        if self.training:
            batch_embed = self.dropout(batch_embed)
        logging.info(batch_embed.shape)
        batch_embed.unsqueeze_(1)
        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            filter_height = sen_len - self.filter_sizes[i] + 1  # 卷积后的 长度
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)
            mp = torch.nn.MaxPool2d((filter_height, 1))
            pooled = mp(hidden).reshape(sen_num, self.out_channel)
            pooled_outputs.append(pooled)

        reps = torch.cat(pooled_outputs, dim=1)
        if self.training:
            reps = self.dropout(reps)
        return reps


sent_hidden_size = 256
sent_num_layers = 2

class SentEncoder(torch.nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.sent_lstm = torch.nn.LSTM(
            input_size=sent_rep_size, # 每个句子经过 CNN 后得到 300 维向量
            hidden_size=sent_hidden_size,# 输出的维度
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        sent_hiddens, _ = self.sent_lstm(sent_reps)  
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)
        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)
        return sent_hiddens


class Model(torch.nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = 300 
        self.doc_rep_size = sent_hidden_size * 2  # 双向LSTM 输出后的长度
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordCNNEncoder(vocab)
        
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)

        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = torch.nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters['basic_parameters'] = parameters

        logging.info('Build model with cnn word encoder lstm sent encoder ...')
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        batch_input1, batch_inputs2, batch_masks = batch_inputs
        



if __name__ == '__main__':
    pass




