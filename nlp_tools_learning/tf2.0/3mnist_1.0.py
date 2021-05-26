#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   3mnist_1.0.py
@Time    :   2021/03/31 09:59:30
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   tf datasets 
'''

import os 
import tensorflow as tf

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255 - 1 

y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

print(x.shape)
print(y.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(512)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


with tf.GradientTape() as tape:
    x = tf.reshape(x, (-1, 28*28))
    out = model(x)
    loss = tf.square(out - y)
    loss = tf.reduce_sum(loss) / x.shape[0]

grads = tape.gradient(loss, model.trainable_variables)

tf.keras.optimizers.Adam().apply_gradients(zip(grads, model.trainable_variables))

print(loss)


