# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regression_customized_loss.py
@ctime  : 2020/9/22 10:02
@Email  : 1131771202@qq.com
@content: 使用自定义api构造一个分类
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

#自定义loss函数，mse代表均方差（差的平方的均值）
def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                      input_shape=x_train.shape[1:]),   #当input_shape不定义的时候，会自动检测到
    keras.layers.Dense(1),
])
model.summary()
model.compile(loss = customized_mse, optimizer='sgd', metrics=['mean_squared_error'])  #metrics是在训练中最后检测的数值
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]


history = model.fit(x_train_scaled, y_train, validation_data = (x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid()
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)


model.evaluate(x_test_scaled, y_test)