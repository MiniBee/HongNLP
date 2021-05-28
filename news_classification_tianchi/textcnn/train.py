#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/05/28 14:44:57
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from base import *
import time 
from sklearn.metrics import classification_report
from utils import get_examples, data_iter
from data_prepare import get_data
from vocab import Vocab
from models import Model, Optimizer


clip = 5 
epochs = 17 
early_stops = 3 
log_interval = 50

test_batch_size = 128 
train_batch_size = 128

save_model = './cnn0.bin'
save_test = '../data/cnn0.csv'


class Trainer():
    def __init__(self, model, vocab, train_data, dev_data, test_data):
        self.model = model 
        self.report = True 
        self.train_data = get_examples(train_data, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        self.dev_data = get_examples(dev_data, vocab)
        self.test_data = get_examples(test_data, vocab)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.target_names = vocab.target_names

        self.optimizer = Optimizer(model.all_parameters)

        self.step = 0 
        self.early_stop = -1 
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training ... ')
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)


    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0 
        losses = 0 
        batch_idx = 1 
        y_pred = []
        y_true = []

        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)


    def batch2tensor(self, batch_data):
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs= torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]  # 表示一个句子
                for word_idx in range(sent_data[0]):
                    batch_inputs[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1 
        
        if use_cuda:
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

        return (batch_inputs, batch_masks), batch_labels



if __name__ == '__main__':
    train_data, dev_data, test_data = get_data(fold_num, dev_fold)
    vocab = Vocab(train_data)
    model = Model(vocab)
    trainer = Trainer(model, vocab, train_data, dev_data, test_data)
    trainer.train()
    print(vocab._label2id)







