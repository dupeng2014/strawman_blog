# -*- coding: utf-8 -*-

import numpy as np

print('使用列表生成一维数组')
data = [1,2,3,4,5,6]
x = np.array(data)
print(x) #打印数组
print(x.dtype) #打印数组元素的类型

print('使用列表生成二维数组')
data = [[1,2],[3,4],[5,6]]
x = np.array(data)
print(x) #打印数组
print(x.ndim) #打印数组的维度
print(x.shape) #打印数组各个维度的长度。shape是一个元组

print('使用zero/ones/empty创建数组:根据shape来创建')
# 默认创建的数组类型(dtype)都是float64。
x = np.zeros(6) #创建一维长度为6的，元素都是0一维数组
print(x)
x = np.zeros((2,3)) #创建一维长度为2，二维长度为3的二维0数组
print(x)
x = np.ones((2,3), dtype=np.int64) #创建一维长度为2，二维长度为3的二维1数组
print(x)
x = np.empty((3,3)) #创建一维长度为2，二维长度为3,未初始化的二维数组
print(x)

print('使用arrange生成连续元素')
print(np.arange(6))  # [0,1,2,3,4,5,] 开区间
print(np.arange(0, 6, 2))  # [0, 2，4]








x = np.arange(12)
print(x)
print('改变数组的形状')
x = x.reshape((3, 4))
print(x)
# 打印数据的形状
print(x.shape)

# 矩阵转置
print(x.T)

# 再转换成1维数组
x_one_1 = x.reshape((12,))
print(x_one_1)
x_one_2 = x.flatten()
print(x_one_2)









print('ndarray数组与标量/数组的运算')
x = np.array([1,2,3,4])
print(x*2) # [2 4 6 8]
print(x>2) # [False False  True True]
y = np.array([3,4,5,6])
print(x+y) # [4 6 8 10]
print(x>y) # [False False False Flase]
x = x.reshape((2,2))
y = y.reshape((2,2))
print(x + y)
print(x * y)
print(np.dot(x, y))









x = np.arange(12)
a = x.reshape(4, 3)
b = x.reshape(3, 4)
# 取第3行
print(b[2])
# 取连续的多行（2～4行）
print(b[1:4])
# 取不连续的多行(1、3行)
print(b[[0, 2]])
# 取第2列
print(b[:, 1])
# 取连续的多列（1～3列）
print(b[:, 0:3])
# 取不连续的多列（1，3列）
print(b[:, [0, 2]])
# 取行列交叉的点（3行4列）
c = b[2, 3]
print(c)
print(type(c))
# 取多行和多列，取第3行到第五行，第2列到第4列的结果
# 去的是行和列交叉点的位置
print(b[2:5, 1:4])
# 取多个不相邻的点
# 选出来的结果是（0，0） （2，1） （2，3）
print(b[[0, 2, 2], [0, 1, 3]])





print('数组的转置和轴对换')
k = np.arange(9) #[0,1,....8]
m = k.reshape((3,3)) # 改变数组的shape复制生成2维的，每个维度长度为3的数组
print(k) # [0 1 2 3 4 5 6 7 8]
print(m) # [[0 1 2] [3 4 5] [6 7 8]]
# 转置(矩阵)数组：T属性 : mT[x][y] = m[y][x]
print(m.T) # [[0 3 6] [1 4 7] [2 5 8]]
# 计算矩阵的内积 xTx
print(np.dot(m,m.T)) # numpy.dot点乘
# 高维数组的轴对象
k = np.arange(8).reshape(2,2,2)
print(k) # [[[0 1],[2 3]],[[4 5],[6 7]]]
print(k[1][0][0])
# 轴变换 transpose 参数:由轴编号组成的元组
m = k.transpose((1,0,2)) # m[y][x][z] = k[x][y][z]
print(m) # [[[0 1],[4 5]],[[2 3],[6 7]]]
print(m[0][1][0])
# 轴交换 swapaxes (axes：轴)，参数:一对轴编号
m = k.swapaxes(0,1) # 将第一个轴和第二个轴交换 m[y][x][z] = k[x][y][z]
print(m) # [[[0 1],[4 5]],[[2 3],[6 7]]]
print(m[0][1][0])
# 使用轴交换进行数组矩阵转置
m = np.arange(9).reshape((3,3))
print(m) # [[0 1 2] [3 4 5] [6 7 8]]
print(m.swapaxes(1,0)) # [[0 3 6] [1 4 7] [2 5 8]]






print('一元ufunc示例')
x = np.arange(6)
print(x) # [0 1 2 3 4 5]
print(np.square(x)) # [ 0  1  4  9 16 25]
x = np.array([1.5,1.6,1.7,1.8])
y,z = np.modf(x)
print(y) # [ 0.5  0.6  0.7  0.8]
print(z) # [ 1.  1.  1.  1.]


print('二元ufunc示例')
x = np.array([[1,4],[6,7]])
y = np.array([[2,3],[5,8]])
print(np.maximum(x,y)) # [[2,4],[6,8]]
print(np.minimum(x,y)) # [[1,3],[5,7]]








print('numpy的基本统计方法')
x = np.array([[1,2],[3,3],[1,2]]) #同一维度上的数组长度须一致
print(x.mean()) # 2
print(x.mean(axis=1)) # 对每一行的元素求平均
print(x.mean(axis=0)) # 对每一列的元素求平均
print(x.sum()) #同理 12
print(x.sum(axis=1)) # [3 6 3]
print(x.max()) # 3
print(x.max(axis=1)) # [2 3 2]
print(x.cumsum()) # [ 1  3  6  9 10 12]
print(x.cumprod()) # [ 1  2  6 18 18 36]



print('sort的就地排序')
x = np.array([[1,6,2],[6,1,3],[1,5,2]])
x.sort(axis=1)
print(x) # [[1 2 6] [1 3 6] [1 2 5]]
#非就地排序：numpy.sort()可产生数组的副本






print('线性代数')
import numpy.linalg as nla
print('矩阵点乘')
x = np.array([[1,2],[3,4]])
y = np.array([[1,3],[2,4]])
print(x.dot(y)) # [[ 5 11][11 25]]
print(np.dot(x,y)) # # [[ 5 11][11 25]]
print('矩阵求逆')
x = np.array([[1,1],[1,2]])
y = nla.inv(x) # 矩阵求逆（若矩阵的逆存在）
print(x.dot(y)) # 单位矩阵 [[ 1.  0.][ 0.  1.]]
print(nla.det(x)) # 求行列式





print('数组的合并与拆分')
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
print(np.concatenate([x, y], axis = 0) )
# 竖直组合 [[ 1  2  3][ 4  5  6][ 7  8  9][10 11 12]]
print(np.concatenate([x, y], axis = 1))
# 水平组合 [[ 1  2  3  7  8  9][ 4  5  6 10 11 12]]
print('垂直stack与水平stack')
print(np.vstack((x, y))) # 垂直堆叠:相对于垂直组合
print(np.hstack((x, y))) # 水平堆叠：相对于水平组合
# dstack：按深度堆叠
print(np.split(x,2,axis=0) )
# 按行分割 [array([[1, 2, 3]]), array([[4, 5, 6]])]
print(np.split(x,3,axis=1) )
# 按列分割 [array([[1],[4]]), array([[2],[5]]), array([[3],[6]])]

# 堆叠辅助类
import numpy as np
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = np.random.randn(3, 2)
print('r_用于按行堆叠')
print(np.r_[arr1, arr2])
'''
[[ 0.          1.        ]
 [ 2.          3.        ]
 [ 4.          5.        ]
 [ 0.22621904  0.39719794]
 [-1.2201912  -0.23623549]
 [-0.83229114 -0.72678578]]
'''
print('c_用于按列堆叠')
print(np.c_[np.r_[arr1, arr2], arr])
'''
[[ 0.          1.          0.        ]
 [ 2.          3.          1.        ]
 [ 4.          5.          2.        ]
 [ 0.22621904  0.39719794  3.        ]
 [-1.2201912  -0.23623549  4.        ]
 [-0.83229114 -0.72678578  5.        ]]
'''
print('切片直接转为数组')
print(np.c_[1:6, -10:-5])
'''
[[  1 -10]
 [  2  -9]
 [  3  -8]
 [  4  -7]
 [  5  -6]]
'''
