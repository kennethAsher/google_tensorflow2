#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_data_basic_api.py
@Time:          2020/9/24 6:09 下午
@Desc:          //TODO 对数据集进行estimator操作，使用tf自带的estimator
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

#对原本的数据做笛卡尔积，更形象的描绘出关系，
# cross feature: age: [1,2,3,4,5], gender:[male, female]
# age_x_gender: [(1, male), (2, male), ..., (5, male), ..., (5, female)]
# 但是会造成数据量太大，需要做hash_bucket_size 也就是所有的hash值在做取余，最终结果保持在100个桶内，大大减小了空间的占用
# 100000: 100 -> hash(100000 values) % 100
# 新添加的交叉特征，需要考虑不能特征对应不同模型的兼容，有的能是准确提高，有的只能降低准确率，如果一定要用的话，建议使用wide&deep模型
feature_columns.append(tf.feature_column.indicator_column(
    tf.feature_column.crossed_column(['age','sex'], hash_bucket_size=100)))

# 构建dataset
def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    # 在这里使用dict能够是输入的data_df的格式能够被from_tensor_slices调用
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
        #repeat 能够重复执行epochs次，将总量变大，batch能够按照这样子的格式定大小输出
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


# 首先需要自定义文件夹， 可以在对应的文件夹中查看TensorBoard
# BaselineClassifier最基本的分类起
output_dir = 'baseline_model'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# 可能是环境问题，一直没有解决
# baseline_estimator = tf.estimator.BaselineClassifier(model_dir=output_dir, n_classes=2)
# 两个参数输出路径和一共拥有的类目数量
baseline_estimator = tf.compat.v1.estimator.BaselineClassifier(model_dir=output_dir, n_classes=2)
# 按照当前分类器去训练
baseline_estimator.train(input_fn = lambda : make_dataset(
    train_df, y_train, epochs = 100))

# 验证分类器
baseline_estimator.evaluate(input_fn=lambda:make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))


linear_output_dir = 'linear_model'
if not os.path.exists(linear_output_dir):
    os.mkdir(linear_output_dir)
# 需要将我们定义好的feature_columns当作参数传入
linear_estimator = tf.estimator.LinearClassifier(model_dir=linear_output_dir, n_classes=2, feature_columns=feature_columns)
linear_estimator.train(input_fn = lambda: make_dataset(train_df, y_train, epochs=100))

linear_estimator.evaluate(input_fn = lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False))

# 深度学习estimator
dnn_output_dir = './dnn_model'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)
#DNNclassifier需要定义寻来呢的层数以及每层的大小，定义激活函数，定义学习的方式
dnn_estimator = tf.estimator.DNNClassifier(
    model_dir = dnn_output_dir, n_classes=2, feature_columns=feature_columns,
    hidden_units=[128,128], activation_fn=tf.nn.relu, optimizer='Adam')
dnn_estimator.train(input_fn = lambda: make_dataset(train_df, y_train, epochs=100))
dnn_estimator.evaluate(input_fn=lambda:make_dataset(eval_df, y_eval, epochs=1, shuffle=False))