# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_keras_classification_model.py
@ctime  : 2020/9/16 13:30
@Email  : 1131771202@qq.com
@content: 创建一个标准的分类模型
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

# 不同物品的数据集
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:8000], x_train_all[8000:]
y_valid, y_train = y_train_all[:8000], y_train_all[8000:]
# 分别获得验证集，训练集，测试集
print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")  # cmap 展示效果，binary指的是二维图
    plt.show()


# show_single_image(x_train[0])

def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data), "使用数据必须是同一类数据，训练集不能和测试机对应"
    assert n_rows * n_cols < len(x_data), "总展示的数据量不能超多数据量总和"
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))  # 设置展示的画布大小
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_data[index], cmap='binary', interpolation='nearest')  # nearest 缩放按照周围最近的比例
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()


class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# show_imgs(3,5,x_train, y_train, class_names)

# 下面来常见一个算法

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28,28]),
#     keras.layers.Dense(300, activation='relu'),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
# 当输入y为一个数值的时候，index，使用sparse， 当输入y为列表，例如onehot的时候，不适用sparse
# optimizer 选择内容很重要
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train_all, y_train_all, epochs=10, validation_split=0.2)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)   # 显示图坐标中的网格
    plt.gca().set_ylim(0, 1)  #设置边界的范围值
    plt.show()

plot_learning_curves(history)
