"""
 3 ###############################################################################
 4 # 作者：wanglei5205
 5 # 邮箱：wanglei5205@126.com
 6 # 代码：http://github.com/wanglei5205
 7 # 博客：http://cnblogs.com/wanglei5205
 8 # 目的：xgboost基本用法
 9 ###############################################################################
10 """
### load module

from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

 ### load datasets
digits = datasets.load_digits()

### data analysis
print(digits.data.shape)  # 输入空间维度
print(digits.target.shape)  # 输出空间维度
### data split

x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size = 0.3,random_state = 33)

### fit model for train data

model = XGBClassifier()
model.fit(x_train, y_train)

### make prediction for test data
y_pred = model.predict(x_test)

### model evaluate
accuracy = accuracy_score(y_test, y_pred)
print("accuarcy: %.2f%%" % (accuracy * 100.0))