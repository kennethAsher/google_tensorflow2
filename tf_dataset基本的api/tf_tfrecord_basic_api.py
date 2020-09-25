#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_tfrecord_basic_api.py
@Time:          2020/9/25 11:01 上午
@Desc:          //TODO 对tfrecord文件格式的基本操作数据类型
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


# TFrecord文件格式：高效存储读取
# 使用tf-record的步骤
#     -> tf.train.example
#         -> tf.train.Features -> {"key":tf.train.Feature}
#             -> tf.train.Feature -> tf.train.ByteList/FloatList/Int64List
# 创建一个Feature
favorite_books = [name.encode('utf8') for name in ['machine learning','cc150']]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)
hours_floatlist = tf.train.FloatList(value=[12.0, 9.5, 7.0, 8.0])
print(hours_floatlist)
age_int64list = tf.train.Int64List(value = [42])
print(age_int64list)

feature = tf.train.Features(
    feature = {
        "favorite_books":tf.train.Feature(bytes_list = favorite_books_bytelist),
        "hours":tf.train.Feature(float_list = hours_floatlist),
        "age":tf.train.Feature(int64_list = age_int64list)
    }
)

print(feature)


#  倒退，根据feature得到example: 将所有的feature组合成一个features
example = tf.train.Example(features=feature)
print(example)
# 然后将其序列化达到见效存储空间的效果
service_example = example.SerializeToString()
print(service_example)


# 输出保存  ---至此，正常的文件保存完成
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
file_name = "test.tfrecords"
file_name_fullpath = os.path.join(output_dir, file_name)
# 将数据保存成tfrecord文件
with tf.io.TFRecordWriter(file_name_fullpath) as writer:
    for i in range(3):
        writer.write(service_example)


# 然后取读取已经保存的tfrecord文件
dataset = tf.data.TFRecordDataset([file_name_fullpath]) #需要读一个目录下的所有文件，所以用list
for serialized_example_tensor in dataset:
    # 都是已经存储好的二进制文件
    print(serialized_example_tensor)

# 将二进制文件倒推出原始数据
expected_features = {
    "favorite_books":tf.io.VarLenFeature(dtype=tf.string),
    "hours":tf.io.VarLenFeature(dtype=tf.float32),
    "age":tf.io.VarLenFeature(dtype=tf.int64)
}
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)   #前面参数是需要解析的tensor，后面参数是对应的类型
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"") # 需要添加一个默认的数值，解析到否则解析到空的时候报错
    for book in books:
        print(book.numpy().decode('utf8'))

# 将record压缩保存，是的数据的存储更省空间
filename_fullpath_zip = file_name_fullpath+'.zip'
options = tf.io.TFRecordOptions(compression_type='GZIP')
# 在创建写出record的时候，需要执行压缩格式
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(service_example)

# 从压缩文件进行读取
dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type="GZIP")  #文件目录，压缩格式
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    # 在example中读取指定的feature
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode('utf8'))

