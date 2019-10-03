import requests
import time



# 程序开始时间
start = time.time()
res = requests.post("http://127.0.0.1:7777/")
# 程序运行所需时间
elapsed = (time.time() - start)
print(res.text)
print('程序运行所需时间：', elapsed)
