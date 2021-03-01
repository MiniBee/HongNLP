#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/02/24 14:40:26
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from src.evaluate import hmm_train_eval
from src.data import build_corpus

def main():
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus('train')
    test_word_lists, test_tag_lists = build_corpus('test', make_vocab=False)
    dev_word_lists, dev_tag_lists = build_corpus('dev', make_vocab=False)

    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id, tag2id
    )


if __name__ == "__main__":
    main()


