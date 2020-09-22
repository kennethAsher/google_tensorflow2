# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_diffs.py
@ctime  : 2020/9/21 13:41
@Email  : 1131771202@qq.com
@content: tf求导
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


# 为什么提供自定义求导（有的官方给出的求导方法并不能解决我们需要用到的复杂的求导方式）
def f(x):
        return 3*x**2+2*x-1
# 近似导数：（（某处的值+一个极小的数）-（某处的值-极小的数））/2倍极小的数
def approximate_derivative(f, x, eps=1e-3):
    return (f(x+eps)-f(x-eps))/(2.*eps)
print(approximate_derivative(f, 1.))



def g(x1, x2):
    return (x1+5)*(x2**2)
def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2
print(approximate_gradient(g, 2., 3.))



#使用GradientTape来求导数，也就是tensorflow自己的求导工具
#默认情况下，初始化的tape只能使用一次，使用第二次会出问题，系统会在试用过一次之后自动释放，类似生成器的规则
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
print(dz_x1)
try:
    dz_x2 = tape.gradient(z, x2)
except RuntimeError as ex:
    print(ex)



# 可以通过设置属性使tape能够多次利用，不够需要自己手动释放掉tape，
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)

print(dz_x1)
try:
    dz_x2 = tape.gradient(z, x2)
    print(dz_x2)
except RuntimeError as ex:
    print(ex)
del tape




# 同时对过个数求导数
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1,x2])

#结果返回的也是列表
print(dz_x1x2)



#都是对能够训练的变量求导，接下来试试对常量求导
#直接怼常量求导是不能到的内容的，返回None，需要我们主动对常量进行监控（watch）
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1,x2])
print(dz_x1x2)



#设置多个求导公式，只输入单个数值的情况下，默认将所有的倒数求和，而不是类似上述统一求导公式，多个数值那样返回列表
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3*x
    z2 = x**2
tape.gradient([z1, z2], x)



x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grads, [x1, x2]) for inner_grad in inner_grads]  # 返回的倒数是一个列表
print(outer_grads)
del outer_tape
del inner_tape



#更新参数
learning_rate = 0.1
x=tf.Variable(0.0)
for _ in range(100):
    with  tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate*dz_dx)
print(x)


#使用tf自带的更新参数，结合keras的optimizer
learning_rate = 0.1
x=tf.Variable(0.0)
optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with  tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    #里面的参数时[(),(),()]
    optimizer.apply_gradients([(dz_dx, x)])
print(x)