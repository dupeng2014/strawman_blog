import requests
import time
import numpy as np


# 程序开始时间
start = time.time()
res = requests.post("http://127.0.0.1:7777/pass_data/aaa")
# 程序运行所需时间
elapsed = (time.time() - start)
print(res.text)
print('程序运行所需时间：', elapsed)



print('---------------------------------------------------------------------')


# 字符串参数的形式传递数据
x_data = np.random.rand(1, 100)
x_data = x_data.tolist()
body = {'data': str(x_data)}

start = time.time()
res = requests.post("http://127.0.0.1:7777/pass_data2/", data=body)
# 程序运行所需时间
elapsed = (time.time() - start)
print(res.text)
print('程序运行所需时间：', elapsed)


print('---------------------------------------------------------------------')


x_data = ['你好', '我是小明', '你是谁']
body = {'data': str(x_data)}

start = time.time()
res = requests.post("http://127.0.0.1:7777/pass_data3/", data=body)
# 程序运行所需时间
elapsed = (time.time() - start)
print(res.text)
print('程序运行所需时间：', elapsed)

