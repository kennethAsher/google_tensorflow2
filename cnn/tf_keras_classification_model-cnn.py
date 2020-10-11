#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_keras_classification_model-cnn.py
@Time:          2020/10/11 3:45 下午
@Desc:          //TODO  实现最基本的卷积神经网络
'''


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

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


#加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all),(x_test, y_test) = fashion_mnist.load_data()
# 切分数据集
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 归一化，使得数值之间的差值减小
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 优先使用fit transform能够使后面在转换的数据格式与当前相同,需要指定通道数量为1
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28,28,1)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1,28,28,1)


# 构建卷积神经网络
# 优先构建模型的外框
model = keras.models.Sequential()
# 使用Conv2D进行卷积网络
#   1.卷积层，对应kernel 进行padding操作，默认每次移动一个单位，产生按照kernel的大小将矩阵内所有的值相加的和产生在新的输出矩阵想，
#     对应的会使产生的矩阵小，设置padding为same的时候，会自动补全0，使产生的结果矩阵大小相同
#   2.通道层数：filter产生为通道数量，每个filter代表每个特征
#   3.polling层，设置pool的大小，在大小的矩阵内选择平均值，或者最大值作为特征进入到下一层

model.add(keras.layers.Conv2D(filters=32,  # filters为输出的通道数量，也是kernel的通道数，也是计算的特征数量
                              kernel_size=3,  # 为采集矩阵的大小
                              padding='same',  # padding的形式，是same为采用补全矩阵，
                              activation='relu',  # 激活函数的方式
                              input_shape=(28, 28, 1)))  # 输入函数的矩阵
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))  # 池化层，将产生的矩阵缩小
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))

model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Flatten())  # 对最终产生的数据集进行压平操作，进行后面相连全联接层，实现分类
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])



logdir='./cnn-selu-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'fashion_mnist.model.h5')
# 回调函数
callbacks=[keras.callbacks.TensorBoard(logdir),
          keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
          keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]  # patience是实现的次数，min_delta是变化值得大小
history = model.fit(x_train_scaled, y_train, epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid()
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)