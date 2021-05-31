#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2021/05/24 14:19:37
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''



import random 
import numpy as np 
import torch
import logging

import sys 
sys.path.append('./')

logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S'
)

seed = 11 
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()

if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu)
else:
    device = torch.device('cpu')


data_file = '/Users/peihongyue/phy/project/HongNLP/news_classification_tianchi/data/train_sample.csv'        # data_prepare 
test_data_file = '/Users/peihongyue/phy/project/HongNLP/news_classification_tianchi/data/test_sample.csv'    # data_prepare
fold_num = 10                                 # data_prepare
dev_fold = 9                                  # data_prepare

min_count = 5                                 # vocab

dropout = 0.15                                # model
word_dims = 100                               # model


