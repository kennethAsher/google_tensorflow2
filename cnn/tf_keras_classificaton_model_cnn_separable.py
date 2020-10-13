#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_keras_classificaton_model_cnn_separable.py
@Time:          2020/10/13 5:41 下午
@Desc:          //TODO 使用深度可分离卷积神经网络执行，比较优缺点
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sklearn
import sys
import time
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all),(x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 标准化，对数值进行缩放，使得所有的数值差不是特别大
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1, 28, 28, 1)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)

#创建一个深度可分离卷积神经网络：
#   深度理解就是对参数进行挑选，获得更少的参数，使得准确率下降，但是空间更小，适用于在手机上执行
#使用方法如下
model = keras.models.Sequential()
# 使用第一层卷积网络去接受数据
# 归一化：使得参数的均值为0，方差为1；批归一化，使得每个层次的参数都进行归一化
model.add(keras.layers.Conv2D(
    filters=32, kernel_size=3, padding='same',
    activation='selu', input_shape=(28,28,1)))   #此处的selu能够理解批归一化和relu的结合
model.add(keras.layers.SeparableConv2D(
    filters=32, kernel_size=3,
    padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))   # 最大值池话，尺寸为2*2
model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
# 在最后对接全链接层的时候，需要对深度学习的数据进行压平
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='selu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 当需要处理的label/y为标签数值的时候，带有sparse  如果是列表形式，类似onehot编码，就不需要代了
model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])


# 查看参数，能够发现：使用深度分离卷积神经网络的参数 ，对比普通的卷积神经网络参数43万，参数降低了到了18万
model.summary()


logdir = ('./separable_cnn_selu_callbacks')
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'fashion_mnist_model.h5')
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),  # 是否只保存最好的模型
    keras.callbacks.EarlyStopping(patience=5, min_delta=True)
]
history = model.fit(x_train_scaled, y_train, epochs=20, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks)


# 通过展示图像，能够看到，图像的准确率在变化幅度是很慢的
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,3)
    plt.show()
plot_learning_curves(history)
