import numpy as np

# 图像矩阵
X = np.array([[1, 1, 1, 1, 1],
              [-1, 0, -3, 0, 1],
              [2, 1, 1, -1, 0],
              [0, -1, 1, 2, 1],
              [1, 2, 1, 1, 1]], dtype=float)
M = len(X)
N = len(X[0])

# 滤波器
W = np.array([[1, 0, 0],
              [0, 0, 0],
              [0, 0, -1]], dtype=float)
U = len(W)
V = len(W[0])

# 二维卷积
Y = np.zeros_like(X)
for i in range(U - 1, M):
    for j in range(V - 1, N):
        for u in range(U):
            for v in range(V):
                Y[i][j] += W[u, v] * X[i - u, j - v]
Y = Y[U - 1:, V - 1:]
print(Y)
'''
[[ 0. -2. -1.]
 [ 2.  2.  4.]
 [-1.  0.  0.]]
'''
