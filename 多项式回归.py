import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)  # 接下来的代码要区分好X和x
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()



#首先用线性回归的方式，可以看出拟合效果很差
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
y_predict = lin_reg.predict(X)
plt.scatter(x,y)
plt.plot(x,y_predict,color='r')
# plt.show()



x2 = np.hstack([X,X**2]) #这里给样本X再引入1个特征项，现在的特征就有2个
lin_reg2 = LinearRegression()
lin_reg2.fit(x2,y)
y_predict2 = lin_reg2.predict(x2)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r') #绘制的时候要注意，因为x是无序的，为了画出如下图平滑的线条，需要先将x进行排序，y_predict2按照x从的大小的顺序进行取值，否则绘制出的如右下图。
# plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) #设置最多添加几次幂的特征项
poly.fit(X)
x2 = poly.transform(X)
print(x2)
#x2.shape 这个时候x2有三个特征项，因为在第1列加入1列1，并加入了x^2项
from sklearn.linear_model import LinearRegression #接下来的代码和线性回归一致
lin_reg2 = LinearRegression()
lin_reg2.fit(x2,y)
y_predict2 = lin_reg2.predict(x2)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
print(x)
print(y_predict2)
plt.plot(x,y_predict2,color='r')
lin_reg2.coef_,lin_reg2.intercept_ #绘制的图像和预测所得的值和上面完全一致，唯一不同的是lin_reg2.coef_有3个系数，第一个系数值为0
plt.show()