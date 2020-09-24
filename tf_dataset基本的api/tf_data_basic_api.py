#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_data_basic_api.py
@Time:          2020/9/24 6:09 下午
@Desc:          //TODO dataset数据集基本的操作
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


# 数据集是我平常使用的dataset里面的数据集，类似房价，手写数据集等
# 自己定义数据集要使用from_tensor_slices
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
# 能看到dataset的类型
print(dataset)
#遍历数据集里面的每个数据
for item in dataset:
    print(item)

# 1.使用repeat来使数据集遍历多少次   类似epoch
# 2.使用batch来重组dataset        类似batch
dataset = dataset.repeat(5)
# 将dataset数据集遍历的3遍
for item in dataset:
    print(item)
dataset = dataset.batch(12)
#重组了数据集，按照顺序每5个组合成一个新年的数据集
for item in dataset:
    print(item)

# 使用interleave来遍历数据集，对立面每个数据集进行操作
# 将数据集中每5个数据为一组，重新组合成数据集
dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),   #map:对每个数据集进行怎么样的操作
    cycle_length = 5,      #并行的数量，加快速度
    block_length = 5,      #每次提取数据集中进行map的数据个数
)
#会发现第一次去第一个数据集的前5个，第二次取第二个数据集的前5个，等到不够的时候，拿到最多的回到第一个数据集继续追加，
#然后按照5个为一块的数据集继续操作，第一波最后省2个，第二次在每行的[5:10]进行取数据，知道结束
for item in dataset2:
    print(item)

# 利用两个数组组合成数据集
x = np.array([[1,2],[3,4],[4,5]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x,y))
#看到类型是数据集类型
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

# 利用字典组合数据集
dataset4 = tf.data.Dataset.from_tensor_slices({'feature':x,'label':y})
for item in dataset4:
    print(item['feature'].numpy(), item['label'].numpy())
