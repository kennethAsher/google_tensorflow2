# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regression_hp_search.py
@ctime  : 2020/9/18 18:16
@Email  : 1131771202@qq.com
@content: 超参数搜索
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

from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=4)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=6)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#超参数搜索：选择其中之一
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
histories = []
for lr in learning_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=[8]),
        keras.layers.Dense(1)
    ])
    #自己定义学习率的参数optimizer
    optimizer = keras.optimizers.SGD(lr)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=100)
    histories.append(history)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
#展示学习率和结果的关系，学习率小，学习速度比较慢，
for lr, history in zip(learning_rates, histories):
    print("Learning Rate: ",lr)
    plot_learning_curves(history)
