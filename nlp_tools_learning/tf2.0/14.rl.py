#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   14.rl.py
@Time    :   2021/05/26 16:55:47
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import tensorflow as tf 
import numpy as np 
import gym


learning_rate = 0.0002
gamma = 0.98


env = gym.make('CartPole-v1')  # 创建游戏环境
env.seed(2333)
tf.random.set_seed(2333)
np.random.seed(2333)


class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = [] 
        self.fc1 = tf.keras.layers.Dense(128, kernel_initializer='he_normal')
        self.fc2 = tf.keras.layers.Dense(2, kernel_initializer='he_normal')

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.softmax(x, axis=1)
        return x 

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, tape):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + gamma * R 
            loss = -log_prob * R
            with tape.stop_recording():
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = []


def main():
    pi = Policy()
    pi(tf.random.normal((4, 4)))
    pi.summary()
    score = 0.0 
    print_interval = 20 
    returns = []

    for n_epi in range(100):
        s = env.reset()
        with tf.GradientTape(persistent=True) as tape:
            for t in range(501):
                s = tf.constant(s,dtype=tf.float32)
                s = tf.expand_dims(s, axis=0)
                prob = pi(s)
                a = tf.random.categorical(tf.math.log(prob), 1)[0]
                a = int(a)
                s_prime, r, done, info = env.step(a)
                pi.put_data((r, tf.math.log(prob[0][a])))
                s = s_prime
                score += r 
                if n_epi > 1000:
                    env.render()
                
                if done:
                    break
                    
            pi.train_net(tape)
        del tape

        if n_epi%print_interval==0 and n_epi!=0:
            returns.append(score/print_interval)
            print(f"# of episode :{n_epi}, avg score : {score/print_interval}")
            score = 0.0

    env.close()
                


if __name__ == '__main__':
    main()
