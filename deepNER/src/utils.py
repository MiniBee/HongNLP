#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/02/24 11:30:31
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch
import pickle


def save_model(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model,f)


def flatten_list(lists):
    flatten_list = []
    for list_ in lists:
        if type(list_) == list:
            flatten_list.extend(list_)
        else:
            flatten_list.append(list_)
    return flatten_list



