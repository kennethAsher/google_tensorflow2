# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_basic_api.py
@ctime  : 2020/9/19 14:23
@Email  : 1131771202@qq.com
@content:
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

t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# index 索引操作
# print(t)
# print(t[:, 1:])
# print(t[..., 1])

#ops 算数操作
# print(t+10)
# print(tf.square(t))  #平方
# print(t@tf.transpose(t))  # 矩阵相乘， 转置


#numpy conversion
# print(t.numpy())
# print(np.square(t))
# np_t =np.array([[1.,2.,3.],[4.,5.,6.]])
# print(tf.constant(np_t))


#scalars
# t = tf.constant(2.718)
# print(t.numpy())
# print(t.shape)
# print(t)


#strings
# t = tf.constant("cafe")
# print(t)
# print(tf.strings.length(t))
# print(tf.strings.length(t, unit="UTF8_CHAR"))
# print(tf.strings.unicode_decode(t, "UTF8"))


#string array
# t = tf.constant(['cafe','coffee','咖啡'])  #在utf8编码中，中文和英文长度相同，每一个字占用一个长度
# print(tf.strings.length(t, unit="UTF8_CHAR"))  #utf8编码长度
# print(tf.strings.unicode_decode(t, "UTF8")) #utf8编码


#ragged tensor #不规则的tensor
r = tf.ragged.constant([[11,22],[21,22,23],[],[41]])
# print(r)
# print(r[1]) #这是拿到一个Tensor张量
# print(r[1:2])  #这样子截取是拿到ragged tensor类型


#ops on ragged tensor
r2 = tf.ragged.constant([[51,52],[],[71]])
print(tf.concat([r, r2], axis=0))


r3 = tf.ragged.constant([[13,14],[15],[],[42,43]])
# print(tf.concat([r,r3], axis=1))


# print(r.to_tensor())  #将不规则的tensor编程普通类型的tensor   自动 填充成一个等长等宽的矩阵，没有数值的位置，按照0进行填充，0所在的位置是每行有数值的后面

#sparse tensor 与不规则tensor不同的是，本tensor能够指定数值的位置
s = tf.SparseTensor(indices=[[0,1],[1,0],[2,3]],     #元素放置的位置
                    values = [1.,2.,3.],             #元素的值
                    dense_shape=[3,4])               #张量的形状
# print(s)
# print(tf.sparse.to_dense(s))  #将sparse tensor转成普通的tensor


#ops on sparse tensors
s2 = s*2.0
# print(s2)
# try:
#     s3 = s+ 1                 #稀疏张量不能直接与数值相加
# except TypeError as ex:
#     print(ex)                 #
s4 = tf.constant([[10.,20.], [30., 40.], [50., 60.], [70., 80.]])
# print(s4)
# print(tf.sparse.sparse_dense_matmul(s, s4))  #sparse tensor 矩阵相乘的方式


#sparse tensor    #sparse tensor必须使用排好序的，不然todense会报错；
s5 = tf.SparseTensor(indices=[[0,2],[0,1],[2,3]], values=[1.,2.,3.], dense_shape=[3,4])
# print(s5)
s6 = tf.sparse.reorder(s5)  #出现必然情况的时候，需要使用reorder来解决倒序的问题
# print(tf.sparse.to_dense(s6))


#Variable
v = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
# print(v)          #变量
# print(v.value())  #内容，变成tensor
# print(v.numpy())  #只有数值  变成矩阵


#assign value    #使用assign去替换，重新赋值
v.assign(2*v)    #对变量重新赋值
print(v.numpy())
v[0,1].assign(42) #对位置从新赋值
print(v.numpy())
v[1].assign([7.,8.,9.])  #对行重新复制
print(v.numpy())

#直接使用=赋值的话会出问 题

