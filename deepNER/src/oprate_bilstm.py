#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   oprate_bilstm.py
@Time    :   2021/03/04 11:38:55
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from copy import deepcopy

from src.config import LSTMConfig, TrainingConfig
from src.utils import sort_by_length, tensorized
from models.BILSTM import BiLSTM, cal_loss



class BiLSTM_operator(object):
    def __init__(self, vocab_size, out_size, crf=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size

        self.crf = crf 
        if crf:
            pass 
        else:
            self.model = BiLSTM(vocab_size, self.emb_size, self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_loss

        self.epochs = TrainingConfig.epochs
        self.lr = TrainingConfig.lr
        self.print_step = TrainingConfig.print_step
        self.batch_size = TrainingConfig.batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.step = 0 
        self._best_val_loss = 1e20
        self.best_model = None

    def train(self, word_lists, tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id):
        word_lists, tag_lists, _ = sort_by_length(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_length(dev_word_lists, dev_tag_lists)

        print('训练数据总量：{}'.format(len(word_list)))
        epoch_iterator = trange(1, self.epochs + 1, desc='Epoch')
        for epoch in epoch_iterator:
            self.step = 0 
            losses = 0. 
            for idx in trange(0, len(word_lists), self.batch_size, desc='Iteration'):
                batch_sents = word_lists[idx:idx+self.batch_size]
                batch_tags = tag_lists[idx:idx+self.batch_size]
                losses += self.train_step(batch_sents, batch_tags, word2id, tag2id)
                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // self.batch_size + 1)
                    print('Epoch: {}, step/total_step: {}/{} {:.2f}% Loss: {:.4f}'.format(epoch, self.step, total_step, 100. * self.step / total_step, losses / self.print_step))
                    losses = 0.
            val_loss = 0.

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1 
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        targets, _ = tensorized(batch_tags, tag2id)
        tensorized_sents, targets = tensorized_sents.to(self.device), targets.to(self.device)

        scores = self.model(tensorized_sents, targets)
        self.model.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0. 
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1 
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]

                tensorized_sents, lengths = tensorized(batch_sents, word2id)
                targets, _ = tensorized(batch_tags, tag2id)
                tensorized_sents, targets = tensorized_sents.to(self.device), targets.to(self.device)

                scores = self.model(tensorized_sents, targets)

                loss = self.cal_loss_func(scores, targets, tag2id)
                
                val_losses += loss
            val_loss = val_losses / val_step
            if val_loss < self._best_val_loss:
                print('保存模型 。。。 ')
                self.best_model = deepcopy(self.model)

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        word_lists, tag_lists, indices = sort_by_length(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        targets, _ = tensorized(tag_lists, tag2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(tensorized_sents, lengths, tag2id)

        


                


