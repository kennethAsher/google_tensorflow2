# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_regression_hp_search.py
@ctime  : 2020/9/18 18:16
@Email  : 1131771202@qq.com
@content: 使用sklearn进行超参数搜索
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
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=4)
x_train ,x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#使用RandomizedSearchCV进行超参数搜索
#1.转化为sklearn模式的model

def build_model(hidden_layers = 1, layer_size = 30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu', input_shape=[8]))
    for _ in range(hidden_layers-1):
        model.add(keras.layers.Dense(layer_size, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer=keras.optimizers.SGD(learning_rate) #自定义学习率
    model.compile(loss='mse', optimizer=optimizer)  #mse均方差，与mean_squared_error相同
    return model
#sklearn模式的model
sklearn_model = KerasRegressor(build_fn = build_model)
history = sklearn_model.fit(x_train_scaled, y_train, epochs=30, validation_data=(x_valid_scaled, y_valid))

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)
#2.定义参数集合
from scipy.stats import reciprocal
param_distribution={
    "hidden_layers":[1,2,3,4],
    "layer_size":np.arange(1,100),
    "learning_rate": reciprocal(1e-4, 1e-2)
}

#3.搜索参数
from sklearn.model_selection import RandomizedSearchCV
#超参数搜索机制，cross_validation：训练集分成n份，n-1份训练，1份验证
#这个方法是搜索超参数的
random_search_cv = RandomizedSearchCV(sklearn_model, param_distribution, n_iter=10, cv=3, n_jobs=1) #n_iter生成多少个参数集合，n_jobs多少个并行处理
#这个方法是训练模型的
random_search_cv.fit(x_train_scaled, y_train, epochs=30, validation_data=(x_valid_scaled, y_valid))



print(random_search_cv.best_params_)  #最好的参数
print(random_search_cv.best_score_)  #最好的分值
print(random_search_cv.best_estimator_)  #最好的model

#获取model
model = random_search_cv.best_estimator_.model
print(model.evaluate(x_test_scaled, y_test))