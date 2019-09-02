# -*- coding: UTF-8 -*-

import numpy as np # 快速操作结构数组的工具
import pandas as pd # 数据分析处理工具


# 样本数据集，第一列为x1，第二列为x2，第三列为分类（二种类别）
data=[
    [-0.017612,14.053064,0],
    [-1.395634,4.662541,1],
    [-0.752157,6.538620,0],
    [-1.322371,7.152853,0],
    [0.423363,11.054677,0],
    [0.406704,7.067335,1],
    [0.667394,12.741452,0],
    [-2.460150,6.866805,1],
    [0.569411,9.548755,0],
    [-0.026632,10.427743,0],
    [0.850433,6.920334,1],
    [1.347183,13.175500,0],
    [1.176813,3.167020,1],
    [-1.781871,9.097953,0],
    [-0.566606,5.749003,1],
    [0.931635,1.589505,1],
    [-0.024205,6.151823,1],
    [-0.036453,2.690988,1],
    [-0.196949,0.444165,1],
    [1.014459,5.754399,1]
]


#生成X和y矩阵
dataMat = np.mat(data)
y = dataMat[:,2]   # 类别变量
b = np.ones(y.shape)  # 添加全1列向量代表b偏量
X = np.column_stack((b, dataMat[:,0:2]))  # 特征属性集和b偏量组成x
X = np.mat(X)

# 特征数据归一化
# import sklearn.preprocessing as preprocessing   #sk的去均值和归一化
# scaler=preprocessing.StandardScaler()
# X = scaler.fit_transform(X)   # 对特征数据集去均值和归一化，可以加快机器性能
# X = np.mat(X)
# # print(X)
# ========逻辑回归========

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print('逻辑回归模型:\n',model)
# 使用模型预测
predicted = model.predict(X)   #预测分类
answer = model.predict_proba(X)  #预测分类概率
print(answer)
print(model.predict(X))


import matplotlib.pyplot as plt

# 绘制边界和散点
# 先产生x1和x2取值范围上的网格点，并预测每个网格点上的值。
h = 0.02
x1_min, x1_max = X[:,1].min() - .5, X[:,1].max() + .5
x2_min, x2_max = X[:,2].min() - .5, X[:,2].max() + .5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
testMat = np.c_[xx1.ravel(), xx2.ravel()]   #形成测试特征数据集
testMat = np.column_stack((np.ones(((testMat.shape[0]),1)),testMat))  #添加第一列为全1代表b偏量
testMat = np.mat(testMat)
Z = model.predict(testMat)

# 绘制区域网格图
Z = Z.reshape(xx1.shape)
plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.Paired)


# 绘制散点图 参数：x横轴 y纵轴，颜色代表分类。x图标为样本点，.表示预测点
plt.scatter(X[:,1].flatten().A[0], X[:,2].flatten().A[0],c=y.flatten().A[0],marker='x')   # 绘制样本数据集
plt.scatter(X[:,1].flatten().A[0], X[:,2].flatten().A[0],c=predicted.tolist(),marker='.') # 绘制预测数据集

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()