# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regresion.py
@ctime  : 2020/9/17 16:46
@Email  : 1131771202@qq.com
@content: 创建一个回归模型，
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

# 输出查看样本
import pprint
pprint.pprint(housing.data[:5])
pprint.pprint(housing.target[:5])

from sklearn.model_selection import train_test_split
# random_state随机种子，随便一个值即可，  在参数中存在test_size, train_size 数值为(0,1)，前者为test站的比例，后者为train的比例，默认是test_size=0.25, train_size=0.75
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 在train那里使用fit，在此时拿到均值和方差，供下面使用，在这里不需要转数据类型和reshspe，因为此时的数据类型是相同的
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=x_train.shape[1:]))
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dense(1))
# TODO：方差均值
model.compile(loss='mean_squared_error', optimizer='sgd')
# 当把min_delta设置的比较大的时候，会提前停止，patience指的是幅度变化不大的次数，min_delta指的是数值，当时n此没有超过这个数值的时候，就会停止
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
history = model.fit(x_train_scaled, y_train, epochs=100, callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history).plot(figsize=(10, 5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history.history)