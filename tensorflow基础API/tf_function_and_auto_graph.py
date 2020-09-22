# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author : kennethAsher
@fole   : tf_function_and_auto_graph.py
@ctime  : 2020/9/22 15:14
@Email  : 1131771202@qq.com
@content: 操作tf.function 和 图结构的操作
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

# tf的function和auto-graph（图结构）
# 正经的python方法
def scaled_elu(z, scale=1.0, alpha=1.0):
    #z>=0: scale*z : scale*alpha*tf.nn.elu(z)    三项表达式
    is_positive = tf.greater_equal(z, 0.0)
    return scale*tf.where(is_positive, z, alpha*tf.nn.elu(z))  #tf的三线表达式
print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant(2.)))

#将普通的python方法转成tf的方法（图）
#专门供应tf使用，优点速度快
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant(-2.)))

#对比是不是同一类型方法
print(scaled_elu_tf.python_function is scaled_elu)  #使用python_funciton  能够将tf 的图方法转成python的方法

#对比图方法和Python方法的执行速度
# %timeit scaled_elu_tf(tf.random.normal((1000,1000)))
# %timeit scaled_elu(tf.random.normal((1000,1000)))

#用另外一种方法，将普通的python方法转变成为tf的方法
#实现1+1/2+1/2^2+。。。+1/2^n
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total
print(converge_to_2(20))

#展示tf.function的图（展示方法的内部结构）
def display_tf_code(func):
    code = tf.autograph.to_code(func)
    # from Ipython.display import display, Markdown
    # display(markdown('```python\n{}\n```'.format(code)))
display_tf_code(scaled_elu_tf)

#不能再tf的图方法里面创建新的tf张量，只能在外部创建，然后在里面调用
var = tf.Variable(0.)

@tf.function
def add_21():
    return var.assign_add(21)  #+=
print(add_21())

#指定tf的图方法的输入参数
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z,3)
try:
    print(cube(tf.constant([1.,2.,3.])))  #shape的能使用小括号，但是数值的只能用列表， 传入的值不符合类型，会出现结果
except ValueError as ex:
    print(ex)
print(cube(tf.constant([1,2])))

# @tf.function: python func -> tf graph
# get_concrete_function: add input signature -> savedmodel
# 通过tf.function能够将Python方法转成tf的图架构方法，使用get_concrete_function能够将添加的signature转变成为可存储的模型
cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
print(cube_func_int32)   #此时是一份concretefunction

print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1,2,3])))

print(cube_func_int32.graph)  #  此时是FuncGraph 将对象实力化，能够调用实例化对象里面的操作

# 展示具体的图结构里面包含了哪些操作
cube_func_int32.graph.get_operations()

# 查看第3步操作是什么
pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)

print(list(pow_op.inputs))
print(list(pow_op.outputs))

#根据名字查看每个步骤的输入和输出具体情况
cube_func_int32.graph.get_operation_by_name("x")
cube_func_int32.graph.get_operation_by_name("Pow/y")

#查看所有步骤的名称
cube_func_int32.graph.as_graph_def()

