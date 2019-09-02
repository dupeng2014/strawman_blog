import numpy as np

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)   #如果deriv为true，求导数
    return 1 / (1 + np.exp(-x))

X = np.array([[0.35],[0.9]])   #输入层
y = np.array([[0.5]])          #输出值

np.random.seed(1)

W0 = np.array([[0.1,0.8],[0.4,0.6]])
W1 = np.array([[0.3,0.9]])
print('original ',W0,'\n',W1)

for j in range(100):
    # 相当于文章中x0
    l0 = X
    # 相当于文章中y1
    l1 = nonlin(np.dot(W0,l0))
    # 相当于文章中y2
    l2 = nonlin(np.dot(W1,l1))

    l2_error = y - l2
    Error = 1 / 2.0 * (y - l2) ** 2
    print("Error:", Error)
    l2_delta = l2_error * nonlin(l2, deriv=True)    ##this will backpack
    # print('l2_delta=',l2_delta)

    # 反向传播
    l1_error = l2_delta * W1
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # 修改权值
    W1 += l2_delta*l1.T
    W0 += l0.T.dot(l1_delta)
    print(W0, '\n', W1)

