# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import lstm_model

# 设计模型
word_size = 128
maxlen = 32

# with open('data/msr_train.txt', 'rb') as inp:
#     texts = inp.read().decode('gbk')
# s = texts.split('\r\n')  # 根据换行切分
with open('data/demo_train.txt', 'rb') as inp:
    texts = inp.read().decode('utf-8')
s = texts.split('\r\n')  # 根据换行切分



def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


s = u''.join(map(clean, s))
# 分句
s = re.split(u'[，。！？、]/[bems]', s)

data = []  # 生成训练样本
label = []


def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))

d['data'] = data
d['label'] = label
# 删除大于maxlen的句子
d = d[d['data'].apply(len) <= maxlen]

d.index = range(len(d))


tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})

# 统计所有字，跟每个字编号
chars = []
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars) + 1)


# 保存数据
import pickle

with open('model/chars.pkl', 'wb') as outp:
    pickle.dump(chars, outp)
print('** Finished saving the data.')

# 生成适合模型输入的格式
from keras.utils import np_utils


# 将字转换为字的编号(x)
# print(chars[['的','是']])
# d['x'] = d['data'].apply(lambda x: print(chars[x]))
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))

# 将y值转为onehot编码，因为y是besmx五个字符，所以采用的one-hot编码的数组为5维
def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].values.reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)

d['y'] = d['label'].apply(trans_one)


def train_bilstm():
    """
    train bilistm
    :return:
    """
    print("start train bilstm")
    model = lstm_model.create_model(maxlen, chars, word_size)
    batch_size = 1024
    history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size,
                        epochs=20, verbose=2)
    # model.save('model/model.h5')
    model.save('model/demo_model.h5')

train_bilstm()
