#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_data_basic_api.py
@Time:          2020/9/24 6:09 下午
@Desc:          //TODO 对数据集进行操作，以及对一些model的训练
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

# 数据都是测试数据的泰坦尼克号获救人员的数据
train_file = './data/titanic/train.csv'
eval_file = './data/titanic/eval.csv'
# 将数据读成pandas格式
train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

print(train_df.head(10))
print(eval_df.head(10))

# 提取标签与数据集
y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

print(train_df.head())
print(eval_df.head())
print(y_train.head())
print(y_eval.head())


# 查看训练数据的详情
train_df.describe()

# bins是将柱状图分成多少条，hist是柱状图
train_df.age.hist(bins=20)

# 不同性别的数量分布
train_df.sex.value_counts().plot(kind='barh')

# 不同班次的数量分布
train_df['class'].value_counts().plot(kind='barh')

# 将数据组合并且按照性别和获救做分组数量分布
pd.concat([train_df, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh')

# 有分类的行
categorical_columns=['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
# 有数字的行
numeric_clomns = ['age', 'fare']
# 类别行
feature_columns = []
for categorical_column in categorical_columns:
    # 表示分类内容包含多少类目
    vocab = train_df[categorical_column].unique()
    print(categorical_column, vocab)
    #统一添加到列中
    feature_columns.append(
        #将分类变成onehot编码的格式
        tf.feature_column.indicator_column(
            #对分类行进行处理
        tf.feature_column.categorical_column_with_vocabulary_list(
            categorical_column, vocab)))

for categorical_column in numeric_clomns:
    #对连续值的行直接shying就可以，需要制定一个数值的类型
    feature_columns.append(tf.feature_column.numeric_column(categorical_column, dtype=tf.float32))

# 构建dataset
def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    # 在这里使用dict能够是输入的data_df的格式能够被from_tensor_slices调用
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
        #repeat 能够重复执行epochs次，将总量变大，batch能够按照这样子的格式定大小输出
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

train_dataset = make_dataset(train_df, y_train, batch_size=5)
for x, y in train_dataset.take(1):
    print(x, y)

# keras.layers.DenseFeature 使用这个方法能够只拿到值
for x, y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_column = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column)(x).numpy())
    print(keras.layers.DenseFeatures(gender_column)(x).numpy())


# keras.layers.DenseFeature
for x, y in train_dataset.take(1):
    # 将feature_columns全部变成onehot编码
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())



model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',      #常量交叉分类
             optimizer=keras.optimizers.SGD(lr=0.01),
             metrics=['accuracy'])


# 1. model.fit
# 2. model-> estimator-> train
train_dataset = make_dataset(train_df, y_train, epochs = 100)
eval_dataset = make_dataset(eval_df, y_eval, epochs = 1, shuffle = False)
history = model.fit(train_dataset,
                    validation_data = eval_dataset,
                    steps_per_epoch = 20,
                    validation_steps = 8,
                    epochs = 100)


# 将model转成estimator
estimator = keras.estimator.model_to_estimator(model)
# 将input_fn设置成一个方法
# 将input_fn返回值为一个元祖，或者dataset(里面是元组的内容)
# 2. return a. (feature, labels) b. dataset -> (feature, label)
# 将 estimator进行训练（下面是没有参数的函数，只是使用lambda来封装一次）
estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))



