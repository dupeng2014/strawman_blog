from flask import Flask
from flask import request
import json


# 初始化一个新的Flask实例
app = Flask(__name__)


# 根据url获取数据
@app.route('/pass_data/<data>', methods=["POST","GET"])
def pass_data(data):
    print(data)
    return str(data)



# 参数的形式
@app.route('/pass_data2/', methods=["POST", "GET"])
def pass_data2():
    data = request.form['data']
    print(data)
    return data


# 将 string类型的data转换成数组
@app.route('/pass_data3/', methods=["POST", "GET"])
def pass_data3():
    data = request.form['data']
    # 将 string类型的data转换成数组
    data = data.replace('(', '[').replace(')', ']').replace('\'', '"')
    data = json.loads(data)
    return str(data)

if __name__ == '__main__':
    # 如果要修改服务对应的IP地址和端口怎么办？
    # 只需要修改这行代码，即可修改IP地址和端口：
    app.run(host='127.0.0.1', port=7777)