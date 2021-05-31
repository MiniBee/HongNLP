#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   task3.py
@Time    :   2021/05/31 17:12:43
@Author  :   Hongyue Pei 
@Version :   1.0
@Desc    :   tf idf 使用，sklearn 文本分类
'''

# here put the import lib
import pandas as pd 
import numpy as np
from sklearn import model_selection 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
import logging

logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S'
)


def get_data():
    logging.info('get data ... ')
    df = pd.read_csv('E:\\project\\data\\news_classification_tianchi\\train_set.csv', sep='\t')
    return df 


def score(y, pred):
    f1 = f1_score(y, pred, average='macro')
    precision = precision_score(y, pred, average='macro')
    recall = precision_score(y, pred, average='macro')
    print(f1, precision, recall)


def vectorizer(df, type='count'):
    logging.info('text vectorizer ... ')
    if type == 'count':
        vectorizer = CountVectorizer(max_features=5000)
    elif type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    vectorizer.fit(df['text'])
    return vectorizer


def classifier(df, vectorizer):
    train_text = vectorizer.transform(df['text'])
    train_y = df['label'].values
    model = RidgeClassifier()
    logging.info('training ... ')
    model.fit(train_text, train_y)
    logging.info('predicting ... ')
    pred_y = model.predict(train_text)
    score(train_y, pred_y)


if __name__ == '__main__':
    df = get_data()
    vectorizer = vectorizer(df, 'count')
    classifier(df, vectorizer)







