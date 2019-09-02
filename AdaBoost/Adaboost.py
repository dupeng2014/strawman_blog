from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append (float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

datArr, labelArr = loadDataSet('data_Adaboost.txt')  # 这里使用了上面获取数据的函数得到数据集和目标变量
clf = AdaBoostClassifier(n_estimators=10)  # 指定10个弱分类器
label = np.array(labelArr)
scores = cross_val_score(clf, np.mat(datArr), label)  # 模型 数据集 目标变量
print(scores.mean())


