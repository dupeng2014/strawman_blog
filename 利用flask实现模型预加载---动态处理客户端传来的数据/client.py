import requests
import time
import numpy as np
import json

print('---开始....')
start = time.time()
feature_number = 100
x_input = np.random.rand(1, feature_number)
x_input = x_input.tolist()
dicJson = json.dumps(x_input)

# 根据不同的 model 号，请求不同的模型
body = {
    'data': dicJson,
    'model': 2
}

res = requests.post("http://127.0.0.1:7778/predict/", data=body)

# res = requests.post("http://191.167.20.249:7777/predict/")
elapsed = (time.time() - start)
res_data = json.loads(res.text)
print(res_data)
print(len(res_data))
print('程序运行所需时间：', elapsed)



