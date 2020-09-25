#!/usr/bin/python
# encoding: utf-8
'''
@Author:        kennethAsher
@Contact:       1131771202@qq.com
@ClassName:     tf_data_generate_csv.py
@Time:          2020/9/25 9:59 上午
@Desc:          //TODO  对数据进行csv读取写出操作
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

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


# 上面是加载的数据，下面是自己建立的dataset数据集
# 判断是否存在路径，没有就新建
output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 输出路径，数据， 文件种类， 开头行， 输出文件数量
def save_to_csv(output_dir, data, name_prefix, header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")   # {:02d} 输入的类型是int类型的，并且默认补充为两位
    file_names = []
    #最后得到n_parts组数据
    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(output_dir, file_idx)
        file_names.append(part_csv)
        with open(part_csv, 'wt', encoding='utf8') as f:
            if header is not None:
                f.write(header+"\n")
            for row_index in row_indices:
                #repr是原生字符串，包括‘’都能做成字符串
                f.write(",".join([repr(col) for col in data[row_index]]))
                f.write('\n')
    return file_names
# np.c_是为了合并数组
train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names+['MidianHouseValue']   #拿到housing数据的header
header_str = ",".join(header_cols)
train_filenemas = save_to_csv(output_dir, train_data, "train", header_str, n_parts=20)



# 一种格式化的标准的输出
import pprint
print('train filenames')
print(train_filenemas)
pprint.pprint(train_filenemas)


# 1.将filename 变成 dataset
# 2.read file -> dataset -> datasets -> merge
# 3.parse csv   解析csv
filename_dataset = tf.data.Dataset.list_files(train_filenemas)   #用于将列表变成dataset
for file_name in filename_dataset:
    print(file_name)


n_readers = 5
# 如果不设置block_length 默认按照处理返回一个大的dataset
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1), #TextLineDataset默认处理文本的dataset，放入路径即可， skip表示跳过的行数，一般跳过一行，也就是跳过header
    cycle_length = n_readers)
#take：获取dataset里面的数据行
for line in dataset.take(10):
    print(line.numpy())  #通过tensor只拿到数值大小


# tf.io.decode_csv(str, record_defaults)  将csv记录转化为张量，每一条记录映射一个张量
# 注意，str必须按照“,”分割，并且数量要与record_defaults的数量对应，空或者str的数量多，都会出错
sample_str = '1,2,3,4,5'
record_defaults = [tf.constant(0, dtype=tf.int32), 0, np.nan, "hello", tf.constant([])]
parsed_fileds = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fileds)


# 对每行操作返回一个元组，
def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)]*n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y
parse_csv_line(b'-0.9868720801669367,0.832863080552588,-0.18684708416901633,-0.14888949288707784,-0.4532302419670616,-0.11504995754593579,1.6730974284189664,-0.7465496877362412,1.138',
               n_fields=9)


# 1. filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv
# 合并在一起做一个完整的
def csv_reader_dataset(filenames, n_reader=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)   #将列表变成dataset
    dataset = dataset.repeat()
    # interleave 对dataset进行操作
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),   #通过路径名称，读取文件
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size)   #打乱顺序的次数
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)  #和interleave操作相同，只不过不会最后进行合并，只是单独的对每行进行操作
    dataset = dataset.batch(batch_size)  #batch多少行为一组
    return dataset
train_set = csv_reader_dataset(train_filenemas, batch_size=3)
for x_batch ,y_batch in train_set.take(2):
    print("x:")
    pprint.pprint(x_batch)
    print("x:")
    pprint.pprint(y_batch)


'''全套代码


from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state = 11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)
output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(output_dir, data, name_prefix,
                header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    
    for file_idx, row_indices in enumerate(
        np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join(
                    [repr(col) for col in data[row_index]]))
                f.write('\n')
    return filenames

train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, "train",
                              header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, "valid",
                              header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, "test",
                             header_str, n_parts=10)
                             
import pprint
print("train filenames:")
pprint.pprint(train_filenames)
print("valid filenames:")
pprint.pprint(valid_filenames)
print("test filenames:")
pprint.pprint(test_filenames)

# 1. filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv

filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)
    
n_readers = 5
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    cycle_length = n_readers
)
for line in dataset.take(15):
    print(line.numpy())
    
# tf.io.decode_csv(str, record_defaults)

sample_str = '1,2,3,4,5'
record_defaults = [
    tf.constant(0, dtype=tf.int32),
    0,
    np.nan,
    "hello",
    tf.constant([])
]
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)

try:
    parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
    
try:
    parsed_fields = tf.io.decode_csv('1,2,3,4,5,6,7', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
    
def parse_csv_line(line, n_fields = 9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

parse_csv_line(b'-0.9868720801669367,0.832863080552588,-0.18684708416901633,-0.14888949288707784,-0.4532302419670616,-0.11504995754593579,1.6730974284189664,-0.7465496877362412,1.138',
               n_fields=9)
               
# 1. filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

train_set = csv_reader_dataset(train_filenames, batch_size=3)
for x_batch, y_batch in train_set.take(2):
    print("x:")
    pprint.pprint(x_batch)
    print("y:")
    pprint.pprint(y_batch)
    
batch_size = 32
train_set = csv_reader_dataset(train_filenames,
                               batch_size = batch_size)
valid_set = csv_reader_dataset(valid_filenames,
                               batch_size = batch_size)
test_set = csv_reader_dataset(test_filenames,
                              batch_size = batch_size)
                              
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=[8]),
    keras.layers.Dense(1),
])
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

history = model.fit(train_set,
                    validation_data = valid_set,
                    steps_per_epoch = 11160 // batch_size,
                    validation_steps = 3870 // batch_size,
                    epochs = 100,
                    callbacks = callbacks)
                    
                    
def plot_garph(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid()   #显示网格
    plt.show()
plot_garph(history)

model.evaluate(test_set, steps = 5160 // batch_size)

'''