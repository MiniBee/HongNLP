#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_prepare.py
@Time    :   2021/05/24 14:17:01
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from base import *
import pandas as pd 


def data2fold(fold_num):
    fold_data = []
    df = pd.read_csv(data_file, sep='\t')
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    total = len(labels)
    idx = list(range(total))
    np.random.shuffle(idx)

    all_texts = []
    all_labels = []
    for i in idx:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = []
        label2id[label].append(i)

    all_idx = [[] for _ in range(fold_num)]
    for label, idxs in label2id.items():
        logging.debug((label, len(idxs)))
        batch_size = int(len(idxs) / fold_num)
        others = len(idxs) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size +1 if i < others else batch_size
            batch_data = [idxs[i * batch_size + b] for b in range(cur_batch_size)]
            all_idx[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = [] 
    other_num = 0 
    start = 0 

    for fold in range(fold_num):
        num = len(all_idx[fold])
        texts = [all_texts[i] for i in all_idx[fold]]
        labels = [all_labels[i] for i in all_idx[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num 
            fold_texts = texts + other_texts[start:end]
            fold_labels = labels + other_labels[start:end]
            start = end 
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_texts)

        idx = list(range(batch_size))
        np.random.shuffle(idx)
        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in idx:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)
    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))
    return fold_data


def split_data(fold_data, dev_fold):
    dev_data = fold_data[dev_fold]

    train_texts = [] 
    train_labels = []
    for i in range(fold_num - 1):
        data = fold_data[i]
        train_texts.extend(data['text'])
        train_labels.extend(data['label'])

    train_data = {'label': train_labels, 'text': train_texts}

    df = pd.read_csv(test_data_file, sep='\t')
    texts = df['text'].tolist()
    test_data = {'label': [0] * len(texts), 'text': texts}

    return train_data, dev_data, test_data


def get_data(fold_num, dev_fold):
    fold_data = data2fold(fold_num)
    train_data, dev_data, test_data = split_data(fold_data, dev_fold)
    logging.debug('splited data ... ')
    logging.debug(train_data['text'][0])
    return train_data, dev_data, test_data

if __name__ == '__main__':
    fold_num = fold_num
    dev_fold = dev_fold
    get_data(fold_num, dev_fold)
    
    















