#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CRF.py
@Time    :   2021/02/24 16:33:49
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch
import torch.nn.functional as F
from sklearn_crfsuite import CRF


class CRFModel(object):
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False):
        self.model = CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        features = [sent2features(sent) for sent in sentences]
        self.model.fit(features, tag_lists)
    
    def test(self, sentences):
        features = [sent2features(sent) for sent in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


def word2features(sent, i):
    word = sent[i]
    prev_word = '<s>' if i == 0 else sent[i-1]
    next_word = '</s>' if i == (len(sent) - 1) else sent[i+1]
    feature = {'w': word, 'w-1': prev_word, 'w-1:w': prev_word + word, 'w+1': next_word, 'w:w+1': word + next_word, 'bias': 1}
    return feature


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]



