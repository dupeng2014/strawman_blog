# 使用sklearn的make_classification生成不平衡数据样本
from sklearn.datasets import make_classification
# 生成一组0和1比例为9比1的样本，X为特征，y为对应的标签
X, y = make_classification(n_classes=2, class_sep=2,
                               weights=[0.9, 0.1], n_informative=3,
                               n_redundant=1, flip_y=0,
                               n_features=20, n_clusters_per_class=1,
                               n_samples=100, random_state=10)



from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
# Counter({0: 90, 1: 10})





from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0) #采用随机过采样（上采样）
x_resample,y_resample=ros.fit_sample(X,y)
print(Counter(y_resample))
# print(x_resample)
# print(y_resample)




from imblearn.under_sampling import RandomUnderSampler
#通过设置RandomUnderSampler中的replacement=True参数, 可以实现自助法(boostrap)抽样
#通过设置RandomUnderSampler中的rratio参数,可以设置数据采样比例
rus=RandomUnderSampler(ratio=0.9,random_state=0,replacement=True) #采用随机欠采样（下采样）
x_resample,y_resample=rus.fit_sample(X, y)
print(Counter(y_resample))







# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X = X.astype('float64')
X_smo, y_smo = smo.fit_sample(X, y)
print(Counter(y_smo))

# 从上述代码中可以看出，SMOTE模型默认生成一比一的数据，如果想生成其他比例的数据，可以使用radio参数。不仅可以处理二分类问题，同样适用于多分类问题
# 可通过radio参数指定对应类别要生成的数据的数量
smo = SMOTE(ratio={1: 30 },random_state=42)
# 生成0和1比例为3比1的数据样本
X_smo, y_smo = smo.fit_sample(X, y)
print(Counter(y_smo))
# Counter({0: 900, 1: 300})




print('anasyn')
# 使用imlbearn库中上采样方法中的ADASYN接口
from imblearn.over_sampling import ADASYN
# 定义SMOTE模型，random_state相当于随机数种子的作用
ana = ADASYN(random_state=42)
X = X.astype('float64')
X_smo, y_smo = ana.fit_sample(X, y)
print(Counter(y_smo))


