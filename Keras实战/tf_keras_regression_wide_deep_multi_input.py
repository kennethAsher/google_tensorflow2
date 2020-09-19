# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regression_wide_deep_multi_input.py
@ctime  : 2020/9/18 17:20
@Email  : 1131771202@qq.com
@content: wide&deep模型多输入
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
print(housing.data.shape)
print(housing.target.shape)

from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=4, train_size=0.8)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=6, train_size=0.8)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_valid_scaled = scaler.transform(x_valid)

#多输入
input_wide = keras.layers.Input(shape=[5])  #wide模型
input_deep = keras.layers.Input(shape=[6])  #deep模型
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_wide, input_deep], outputs=[output])
model.compile(loss = 'mean_squared_error', optimizer='sgd')

x_train_scaled_wide = x_train_scaled[:,:5]
x_train_scaled_deep = x_train_scaled[:,2:]
x_valid_scaled_wide = x_valid_scaled[:,:5]
x_valid_scaled_deep = x_valid_scaled[:,2:]
x_test_scaled_wide = x_test_scaled[:,:5]
x_test_scaled_deep = x_test_scaled[:,2:]
#callbacks EarlyStopping能够使程序提前停止，在训练进步不大的时候，没必要浪费更多的时间去提升微不足道的进步，所以提前终止，得到结果
history = model.fit([x_train_scaled_wide, x_train_scaled_deep], y_train, validation_data=([x_valid_scaled_wide,x_valid_scaled_deep], y_valid), epochs=100)
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)

print(model.evaluate([x_test_scaled_wide, x_test_scaled_deep], y_test))