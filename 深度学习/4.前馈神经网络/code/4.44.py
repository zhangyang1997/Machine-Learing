import numpy as np

# 类别数量
C = 10
# 样本维数
D = 3

# 第0层神经元个数(特征向量的维数)
M_0 = D
# 第1层神经元个数
M_1 = 4
# 第2层神经元个数(类别数量)
M_2 = C


# 第0层到第1层的权重矩阵
W_1 = np.random.rand(M_1, M_0)
# 第0层到第1层的偏置
b_1 = np.ones((M_1, 1))

# 第1层到第2层的权重矩阵
W_2 = np.random.rand(M_2, M_1)
# 第1层到第2层的偏置
b_2 = np.ones((M_2, 1))

# 正则化项，矩阵的F范数
W_F = 0
for i in range(M_1):
    for j in range(M_0):
        W_F += W_1[i][j] ** 2
for i in range(M_2):
    for j in range(M_1):
        W_F += W_2[i][j] ** 2
print(W_F)
'''
17.52260821645134
'''

W_F = np.linalg.norm(W_1) ** 2 + np.linalg.norm(W_2) ** 2
print(W_F)
'''
17.52260821645134
'''
