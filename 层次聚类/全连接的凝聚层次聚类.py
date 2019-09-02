import pandas as pd
import numpy as np

if __name__ == "__main__":
    np.random.seed(1)
    # 设置特征的名称
    variables = ["x", "y", "z"]
    # 设置编号
    labels = ["s1", "s2", "s3", "s4", "s5"]
    # 产生一个(5,3)的数组
    data = np.random.random_sample([5, 3]) * 10
    # 通过pandas将数组转换成一个DataFrame
    df = pd.DataFrame(data, columns=variables, index=labels)
    # 查看数据
    print(df)



    # 获取所有样本的距离矩阵
    # 通过SciPy来计算距离矩阵，计算每个样本间两两的欧式距离，将矩阵矩阵用一个DataFrame进行保存，方便查看

    from scipy.spatial.distance import pdist,squareform
    #获取距离矩阵
    '''
    pdist:计算两两样本间的欧式距离,返回的是一个一维数组
    squareform：将数组转成一个对称矩阵
    '''
    dist_matrix = pd.DataFrame(squareform(pdist(df,metric="euclidean")),
                               columns=labels,index=labels)
    print(dist_matrix)



    # 获取全连接矩阵的关联矩阵
    # 通过scipy的linkage函数，获取一个以全连接作为距离判定标准的关联矩阵(linkage matrix)。

    # 第一列表的是簇的编号，第二列和第三列表示的是簇中最不相似(距离最远)的编号，第四列表示的是样本的欧式距离，最后一列表示的是簇中样本的数量。

    from scipy.cluster.hierarchy import linkage
    #以全连接作为距离判断标准，获取一个关联矩阵
    row_clusters = linkage(dist_matrix.values,method="complete",metric="euclidean")
    #将关联矩阵转换成为一个DataFrame
    clusters = pd.DataFrame(row_clusters,columns=["label 1","label 2","distance","sample size"],
                            index=["cluster %d"%(i+1) for i in range(row_clusters.shape[0])])
    print(clusters)

    # 使用sklearn实现凝聚聚类
    from sklearn.cluster import AgglomerativeClustering
    '''
    n_clusters:设置簇的个数
    linkage：设置判定标准
    '''
    ac = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="complete")
    labels = ac.fit_predict(data)

    # 分为两类 0和1
    print(labels)





