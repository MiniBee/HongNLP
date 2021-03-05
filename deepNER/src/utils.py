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


def sort_by_length(word_lists, tag_lists):
    pairs = zip(word_lists, tag_lists)
    indices = sorted(range(len(pairs)), key=lambda x: len(pairs[x][0]), reverse=True)
    word_lists = [word_lists[i] for i in indices]
    tag_lists = [tag_lists[i] for i in indices]
    return word_lists, tag_lists, indices


def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD

    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = mpas.get(e, UNK)
    lengths = [len(i) for i in batch]
    return batch_tensor, lengths



