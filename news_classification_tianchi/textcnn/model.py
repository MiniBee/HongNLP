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


if __name__ == '__main__':
    pass




