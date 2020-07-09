import numpy as np

'''1.数据预处理'''
# 训练样本数量
N = 4
# 类别数
C = 4
# 特征维度
D = 3
# 学习率
alpha = 0.1
# 特征向量组成的矩阵shape=(N,D)
x_train = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [-7, -8, -9],
                    [10, 11, 12]], dtype=float)
# 类别标签组成的向量
y_train = np.array([1, 2, 0, 3])
# 标签对应的one-hot向量组成的标签矩阵
y_train_v = np.zeros((N, C))
for n in range(N):
    y_train_v[n, y_train[n]] = 1
# 权重向量shape=(C*D)
w = np.ones((C * D), dtype=float) * 0.01

# y.shape=(N,C)
# y.T.shape=(C,N)
# x.shape=(N,D)

y = np.diag(np.ones(C))
# print(y)

for n in range(N):
    y_predict = y[0]
    max = -np.inf
    for c in range(C):
        phi = np.reshape(np.outer(x_train[n], y[c]).T, (C * D))
        temp = np.vdot(w, phi)
        if temp > max:
            max = temp
            y_predict = y[c]
    print(y_predict)
'''
[0. 1. 0. 0.]
[1. 0. 0. 0.]
[0. 1. 0. 0.]
[1. 0. 0. 0.]
'''
