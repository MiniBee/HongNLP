#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   qa.py
@Time    :   2021/02/07 10:13:13
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from bert_serving.client import BertClient

bc = BertClient()

# 利用bert获取句向量
sent_emb = bc.encode(['你吃饭了吗', '你要吃什么东西'])
sent_pair = bc.encode(['first do it ||| then do it right'])
print(sent_emb.shape)
print(sent_emb)
print(sent_pair.shape)

# 获取词向量
vec = bc.encode(['你好', 'bert'])
print(vec.shape)
print(vec)



