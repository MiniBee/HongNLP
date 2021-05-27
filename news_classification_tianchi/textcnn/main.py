#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/05/24 16:16:35
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from base import * 



def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    '''
    text 文章
    vocab 词典
    max_sent_len 每句话的长度
    max_segment 共有多少个句子
    '''
    words = text.strip().split()
    document_len = len(words)
    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)
    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]:index[i+1]]
        assert len(segment) > 0 
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append(segment)

    assert len(segments) > 0 
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[segment_:]
    else:
        return segments


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i*batch_size + b] for b in range(cur_batch_size)]
        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    batched_data = []
    





