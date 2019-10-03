import requests
import time
import numpy as np


x_data = np.array([i/float(5) for i in range(5)])
# x_data = x_data.reshape(1, 5, 1)
x_data = x_data.tolist()
body = {'data': str(x_data)}

start = time.time()
res = requests.post("http://127.0.0.1:7777/predict/", data=body)
# 程序运行所需时间
elapsed = (time.time() - start)
print(res.text)
print('程序运行所需时间：', elapsed)
