import xlrd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
 
# 定义加载数据的函数
def fileload(filename = 'ST1-0-91.xlsx'):
    dataset = []
    workbook = xlrd.open_workbook(filename)
    table = workbook.sheets()[0]
    for row in range(table.nrows):
        dataset.append(table.row_values(row))
    return dataset

# 加载数据
data = fileload()
# list类型转换为array类型
data = np.array(data)
# 选取数据表中的CC、CC1、ZDZH、ZXZH、GGGL、MJ数据，将数据类型设置为float
data = np.array(data[1:, 2:8], dtype=float)
print("data矩阵大小为",data.shape)

# 初始化聚类簇数
n_clusters=2
# 创建kmeans模型：参考内容https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
kmeans = KMeans(n_clusters)
# 调用kmeans中的fit成员函数进行聚类
kmeans = kmeans.fit(data)

#聚类完成，输出信息
s = np.empty(n_clusters)
print("每组数据的类别", kmeans.labels_)
for i in range(0, n_clusters):
    s[i]=np.sum(kmeans.labels_ == i)
    print("第%d类的数据数量%d" % (i, s[i]))
centers = np.around(kmeans.cluster_centers_, decimals=2)#保留两位小数
print("每个类的中心:\n", centers)
print("聚类迭代次数:",kmeans.n_iter_)

#调用kmeans中的predict成员函数对新数据进行预测
test1 = [[2.55, 5.77, 82.30, 37.28, 10.73, 251.36]]
label1 = kmeans.predict(test1)
test2 = [[2.57, 5.94, 80.19, 37.22, 9.1, 209.37]]
label2 = kmeans.predict(test2)
print("新数据test1的类别:",label1)
print("新数据test2的类别:",label2)

# 降维可视化

# print(np.random.randint(0,16448,100))
X_tsne = TSNE(n_components=2).fit_transform(data[0:16648:10,:])
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min) 
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(kmeans.labels_[i]),
             color=plt.cm.Set1(kmeans.labels_[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()