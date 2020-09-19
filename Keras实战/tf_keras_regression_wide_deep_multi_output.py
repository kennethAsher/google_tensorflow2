# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regression_wide_deep_multi_output.py
@ctime  : 2020/9/18 17:52
@Email  : 1131771202@qq.com
@content: 多输出，用于一个模型推测多个结果
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

#切分获得数据
from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=5)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_valid_scaled = scaler.transform(x_valid)

input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)

model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output, output2])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])   #metrics 在输出中显示准确率

x_train_scaled_wide = x_train_scaled[:,:5]
x_train_scaled_deep = x_train_scaled[:,2:]
x_valid_scaled_wide = x_valid_scaled[:,:5]
x_valid_scaled_deep = x_valid_scaled[:,2:]
x_test_scaled_wide = x_test_scaled[:,:5]
x_test_scaled_deep = x_test_scaled[:,2:]

history = model.fit([x_train_scaled_wide, x_train_scaled_deep],[y_train, y_train], validation_data=([x_valid_scaled_wide, x_valid_scaled_deep],[y_valid, y_valid]), epochs=30)

print(model.evaluate([x_test_scaled_wide, x_test_scaled_deep],[y_test, y_test]))

