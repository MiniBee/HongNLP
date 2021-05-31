#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   task2.py
@Time    :   2021/05/31 16:43:54
@Author  :   Hongyue Pei 
@Version :   1.0
@Desc    :   读取数据和分析数据
'''

# here put the import lib
import pandas as pd 
from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv('E:\\project\\data\\news_classification_tianchi\\train_set.csv', sep='\t')
    return df 


def text_len(df):
    df['text_len'] = df['text'].apply(lambda x: len(x.strip().split()))
    print(df['text_len'].describe())
    plt.hist(df['text_len'], bins=200)
    plt.show()


def label_desc(df):
    df['label'].value_counts().plot(kind='bar')
    plt.show()




if __name__ == '__main__':
    df = get_data()
    text_len(df)
    label_desc(df)
    





