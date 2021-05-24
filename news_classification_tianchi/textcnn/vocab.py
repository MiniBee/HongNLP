#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vocab.py
@Time    :   2021/05/24 15:31:40
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from base import * 
from collections import Counter
from transformers import BasicTokenizer


class Vocab():
    def __init__(self, train_data):
        self.min_count = min_count
        self.pad = 0 
        self.unk = 1 
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        
        self._id2label = []
        self.target_names = [] 

        self.build_vocab(train_data)
        reverse = lambda x: dict(zip(x, range(len(x))))
        #创建词和 index 对应的字典
        self._word2id = reverse(self._id2word)
        #创建 label 和 index 对应的字典
        self._label2id = reverse(self._id2label)


    def build_vocab(self, data):
        self.word_counter = Counter()
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)
        
        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}
        self.label_counter = Counter(data['label'])
        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])

    def load_pretrained_embs(self, embfile):
        with open(embfile) as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])
        idx = len(self._id2extword)
        embeddings = np.zeros((word_count + idx), embedding_dim)
        for line in lines[1:]:
            values = line.spilt()
            

if __name__ == '__main__':
    basic_tokenizer = BasicTokenizer()



