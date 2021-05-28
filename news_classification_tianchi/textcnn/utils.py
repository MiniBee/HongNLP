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
from sklearn.metrics import f1_score, precision_score, recall_score



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
        segments.append([len(segment), segment])

    assert len(segments) > 0 
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[segment_:]
    else:
        return segments


def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    label2id = vocab._label2id
    examples = []
    for text, label in zip(data['text'], data['label']):
        id = label2id[label]
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = [] 
        for sent_len, sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            doc.append([sent_len, word_ids])
        examples.append((id, len(doc), doc))
    logging.info('Total %d docs. ' % len(examples))
    return examples


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i*batch_size + b] for b in range(cur_batch_size)]
        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    batched_data = []
    if shuffle:
        np.random.shuffle(data)
        lengths = [example[1] for example in data]
        noisy_lengths = [- (l + np.random.uniform(-noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data 
    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    
    for batch in batched_data:
        yield batch


def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))


def get_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') * 100 
    p = precision_score(y_true, y_pred, average='macro') * 100 
    r = recall_score(y_true, y_pred, average='macro') * 100 
    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)


if __name__ == '__main__':
    pass 




