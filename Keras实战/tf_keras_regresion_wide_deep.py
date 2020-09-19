# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regresion.py
@ctime  : 2020/9/17 16:46
@Email  : 1131771202@qq.com
@content: wide&deep模型，需要使用函数式API
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

# 这个数据不是从keras导入的
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
# train_test_split返回x，x，y，y   loaddata x，y，x，y
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7, train_size=0.8)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11, test_size=0.25)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# 函数式API， 功能API(不能再使用sequential哪种组合式的了)
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
#复合函数：f(x) = h(g(x))，类似于这种函数
concat = keras.layers.concatenate([input, hidden2])  #输入和输出拼接
output = keras.layers.Dense(1)(concat)  #将最后合并的输出拼接最终的输出
model = keras.models.Model(inputs=[input], outputs=[output])  #需要自己固定好model，sequential会自动固定好model
model.compile(loss='mean_squared_error', optimizer='adam')
#出现未执行完停止的情况，这是因为运行状态不达标准，提前终止了
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]
history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)