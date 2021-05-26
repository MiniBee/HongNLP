#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   7.bp_action.py
@Time    :   2021/05/25 11:39:49
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import numpy as np 
import matplotlib.pyplot as plt 


class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1/n_neurons)
        self.bias = bias if bias is not None else np.random.randn(n_neurons) * 0.1
        self.activation = activation
        self.last_activation=None 
        self.error = None 
        self.delta = None 

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1+np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r>0] = 1
            grad[r<=0] =0
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2 
        elif self.activation == 'sigmoid':
            return r*(1-r)
        return r
        

class NeuralNetwork:
    def __init__(self):
        self._layers = []
    
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def feed_forward(self, x):
        for layer in self._layers:
            x = layer.activate(x)
        return x 

    def backpropagation(self, x, y, learning_rate):
        output = self.feed_forward(x)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
            
        for i in range(len(self._layers)):
            layer = self._layers[i]
            o_i = np.atleast_2d(x if i == 0 else self._layers[i-1].last_activation)
            layer.weights += layer.delta * o_i.T * learning_rate

    def train(self, x_train, x_test, y_train, y_test, learning_rate, max_epochs):
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = [] 
        for i in range(max_epochs + 1):
            for j in range(len(x_train)):
                self.backpropagation(x_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(x_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                print('Accuracy: %.2f%%' % (self.accuracy(self.predict(x_test), y_test.flatten()) * 100))
    
    def predict(self, X):
        return self.feed_forward(X)

    def accuracy(self, X, y):
        return np.sum(np.equal(np.argmax(X, axis=1), y)) / y.shape[0]


def train(x_train, x_test, y_train, y_test):
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 50, 'sigmoid'))
    nn.add_layer(Layer(50, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 2, 'sigmoid'))
    nn.train(x_train, x_test, y_train, y_test, 0.01, 50)
    


def get_data(N_SAMPLES, TEST_SIZE):
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=11)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=11)
    print(x_train.shape, y_train.shape)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    N_SAMPLES = 2000 
    TEST_SIZE = 0.3
    x_train, x_test, y_train, y_test = get_data(N_SAMPLES, TEST_SIZE)
    train(x_train, x_test, y_train, y_test)





