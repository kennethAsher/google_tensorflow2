# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regression_customized_loss.py
@ctime  : 2020/9/22 10:02
@Email  : 1131771202@qq.com
@content: 使用自定义api神经的深度，layer进行操作介绍
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11, test_size=0.25)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

layer = tf.keras.layers.Dense(100)  #定义一个layer，输出100的维度
layer = tf.keras.layers.Dense(100, input_shape=(None, 5))  #放在首个的layer， input_shape为输入的矩阵
layer(tf.zeros([10,5]))  #形状


#layer.variables  #含有，kernel，bias两个数值
#x*w+b
layer.trainable_variables   #可训练的参数


# 简单自定义layer   tf.nn.softplus: log(1+e^x)
customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))
print(customized_softplus([-10.,-5.,0.,5.,10.]))


# 自定义layer
class CustomizedDenseLayer(keras.layers.Layer):  # 继承父类
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)  # 实现父类的方法

    def build(self, input_shape):
        """构造所需要的参数, 很多方法都是从父类继承过来直接使用的"""
        self.kernel = self.add_weight(name='kernel', shape=[input_shape[1], self.units],  # 名称和形状
                                      initializer='uniform', trainable=True)  # 初始化的方式和是否可训练
        self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='zeros', trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)


#自定义loss函数，mse代表均方差（差的平方的均值）
def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))
model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu',
                      input_shape=x_train.shape[1:]),   #当input_shape不定义的时候，会自动检测到
    CustomizedDenseLayer(1),
])
model.summary()
model.compile(loss = customized_mse, optimizer='sgd', metrics=['accuracy'])  #metrics是在训练中最后检测的数值
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]


history = model.fit(x_train_scaled, y_train, validation_data = (x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid()
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)


model.evaluate(x_test_scaled, y_test)