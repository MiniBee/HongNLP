#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   envaluate.py
@Time    :   2021/02/24 11:22:28
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import time 
from collections import Counter
import pickle
import os 

from models.HMM import HMM
from models.CRF import CRFModel
from src.utils import save_model
from src.evaluating import Metrics


def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_0=False):
    train_word_lists,train_tag_lists = train_data
    test_word_lists,test_tag_lists = test_data
    model = HMM(len(tag2id), len(word2id))
    model.train(train_word_lists, train_tag_lists, word2id, tag2id)

    save_model(model, '../models/st_models/deepNER/hmm.pkl')

    pred_tag_lists = model.test(test_word_lists,word2id,tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores(dtype='HMM')


def crf_train_eval(train_data, test_data, remove_0=False):
    print('CRF模型评估训练')
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data
    model = CRFModel()
    model.train(train_word_lists, train_tag_lists)
    save_model(model, '../models/st_models/deepNER/crf.pkl')

    pred_tag_lists = model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores(dtype='CRF')


def ensemble_evaluate(results, targets, remove_o = False):
    for i in range(len(results)):
        pass


