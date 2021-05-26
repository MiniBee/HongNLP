#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   5mpg.py
@Time    :   2021/05/11 15:29:56
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   Mile per Gallon 汽车油耗预测
'''


import tensorflow as tf 
import pandas as pd
import logging


logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S'
)


def get_data(data_path):
    columns_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(data_path, names=columns_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
    return raw_dataset


def h_na(dataset):
    dataset = dataset.dropna()
    return dataset


def split_dataset(dataset, frac=0.8, random_state=0):
    train_dataset = dataset.sample(frac=frac, random_state=random_state)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    return train_dataset, train_labels, test_dataset, test_labels


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 


def train(train_db):
    model = Network()
    model.build(input_shape=(None, 9))
    logging.info('\n{}'.format(model.summary()))
    optimizer = tf.keras.optimizers.RMSprop(1e-3)
    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.reduce_mean(tf.keras.losses.MSE(y, out))
                mae_loss = tf.reduce_mean(tf.keras.losses.MAE(y, out))
            if step % 5 == 0:
                logging.info('{}->{}->mse{}'.format(epoch, step, float(loss)))
                logging.info('{}->{}->mae{}'.format(epoch, step, float(mae_loss)))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main(data_path):
    logging.info('reading data ... ')
    data_path = tf.keras.utils.get_file('auto-mpg.data', data_path)
    raw_dataset = get_data(data_path)
    dataset = raw_dataset.copy()
    logging.debug('\n{}'.format(dataset.head()))
    logging.debug('\n{}'.format(dataset.isna().sum()))
    logging.info('处理缺失数据 ... ')
    dataset = h_na(dataset)
    logging.debug('\n{}'.format(dataset.isna().sum()))

    origin = dataset.pop('Origin')
    # 根据 origin 列来写入新的 3 个列
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0

    logging.info('split train test')
    train_dataset, train_labels, test_dataset, test_labels = split_dataset(dataset)
    logging.debug('\n{}'.format(train_dataset.describe()))
    logging.info('归一化。。。 ')
    train_stats = train_dataset.describe()
    # train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)
    logging.debug('\n{}\n{}'.format(train_dataset.shape, normed_train_data.shape))

    train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
    train_db = train_db.shuffle(100).batch(32)

    train(train_db)
    
    

if __name__ == '__main__':
    data_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    main(data_path)









