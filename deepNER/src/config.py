#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2021/03/04 11:43:44
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

class LSTMConfig(object):
    emb_size = 256
    hidden_size = 256

class TrainingConfig(object):
    batch_size = 16 
    lr = 0.0005
    epochs = 5 
    print_step = 100
    


