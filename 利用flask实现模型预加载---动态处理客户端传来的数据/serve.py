from flask import Flask
# Part3: 若不希望重复定义计算图上的运算，可直接加载已经持久化的图

import os
import shutil
import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.ops import control_flow_ops
import json


# 造一些随机输入数据
num_points = 30000  # 总数据条数
feature_number = 100  # 每条输入数据有100个feature

def load_data():
    x_data = np.random.rand(num_points, feature_number)
    y_data = np.random.randint(0, 2, (num_points, 1))
    return x_data, y_data


def generator_fn(x_data, y_data):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    # token2idx, _ = load_vocab(vocab_fpath)
    for x, y in zip(x_data, y_data):
        # x = encode(sent1, "x", token2idx)
        # y = encode(sent2, "y", token2idx)
        # decoder_input, y = y[:-1], y[1:]

        # x_seqlen, y_seqlen = len(x), len(y)
        yield x, y

def input_fn(x_data, y_data, batch_size, shuffle=False):
    '''

    :return:
    '''
    shapes = ([feature_number], [1])
    types = (tf.float32, tf.float32)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(x_data, y_data))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    # dataset = dataset.batch(10)
    dataset = dataset.padded_batch(batch_size, shapes).prefetch(1)

    return dataset


def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)
def get_batch(batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    x_data, y_data = load_data()
    batches = input_fn(x_data, y_data, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(x_data), batch_size)

    return batches, num_batches, len(x_data)



from flask import request, jsonify

app = Flask(__name__)
# 日志配置
logging.basicConfig(filename="app.log")


@app.route('/', methods=["POST","GET"])
def hello_world():
    arr = ['Hello World!','Hello World!']
    return str(arr)



x_actual = tf.placeholder(tf.float32, [None, feature_number], name="myInput")
def add_layer(inputs, input_size, output_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases  # WX + b
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


# 隐层1
l1 = add_layer(x_actual, feature_number, 32, activation_function=tf.nn.relu)
# 隐层2
l2 = add_layer(l1, 32, 64, activation_function=tf.nn.tanh)
# 隐层3
l3 = add_layer(l2, 64, 32, activation_function=tf.nn.relu)
# 输出层
y_predict = add_layer(l3, 32, 1, activation_function=tf.nn.sigmoid)

saver = tf.train.Saver()
print('---预加载完成---')
mySess = tf.Session()


def pred_client_data(sess, data, batche_size=128):
    '''
    分批处理客户端传来的数据
    :param sess:
    :param data:
    :param batche_size:
    :return:
    '''
    hypotheses = []
    num_batches = int(len(data) / batche_size) + 1
    remain_data = data[(num_batches-1)*batche_size:]
    for i in range(num_batches):
        if i < num_batches-1:
            curr_data = data[i*batche_size:(i+1)*batche_size]
        else:
            curr_data = remain_data

        if curr_data:
            feed_dict = {x_actual: curr_data}
            h = sess.run(y_predict, feed_dict)
            hypotheses.extend(h.tolist())
    return hypotheses



model_dic = {
    1: 'model-0-.ckpt',
    2: 'model-500-.ckpt'
}


@app.route('/predict/', methods=["POST", "GET"])
def predict():

    data = request.form['data']
    # json 转换成数组
    x_data = json.loads(data)
    print(len(x_data))
    model_num = int(request.form['model'])
    model_name = model_dic[model_num]
    saver.restore(mySess, "./Model/%s" % model_name)  # 注意路径写法

    res = pred_client_data(mySess, x_data, 128)
    return jsonify(res)



if __name__ == '__main__':

    # 放在linux上时，一定要加上host
    # app.run(host='191.167.20.249', port=7778)

    app.run(host='127.0.0.1', port=7778)

