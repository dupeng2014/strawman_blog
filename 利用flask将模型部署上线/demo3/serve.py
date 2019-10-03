from flask import Flask
from flask import request
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# 初始化一个新的Flask实例
app = Flask(__name__)

@app.route('/predict/', methods=["POST", "GET"])
def predict():
    data = request.form['data']
    # 将 string类型的data转换成数组
    data = data.replace('(', '[').replace(')', ']').replace('\'', '"')
    data = json.loads(data)
    data = np.array(data)
    data = data.reshape(-1, 5, 1)


    result = model.predict(data, batch_size=1, verbose=0)
    result = result.tolist()
    return str(result)

if __name__ == '__main__':

    # 如果要修改服务对应的IP地址和端口怎么办？
    # 只需要修改这行代码，即可修改IP地址和端口：
    model = Sequential()
    model.add(LSTM(5, input_shape=(5, 1)))
    model.add(Dense(5))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.load_weights('./model/model.h5')
    model._make_predict_function()


    app.run(host='127.0.0.1', port=7777)