from flask import Flask


# 初始化一个新的Flask实例
app = Flask(__name__)

@app.route('/', methods=["POST","GET"])
def hello_world():
    arr = ['Hello World!','Hello World!']
    # print(arr)
    return str(arr)

if __name__ == '__main__':
    # 如果要修改服务对应的IP地址和端口怎么办？
    # 只需要修改这行代码，即可修改IP地址和端口：
    app.run(host='127.0.0.1', port=7777)