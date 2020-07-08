import numpy as np


def rot180(W):
    U = len(W)
    V = len(W[0])
    W_rot180 = np.zeros_like(W)
    for u in range(U):
        for v in range(V):
            W_rot180[u][v] = W[U - u - 1, V - v - 1]
    return W_rot180


# 图像矩阵
X = np.array([[1, 1, 1, 1, 1],
              [-1, 0, -3, 0, 1],
              [2, 1, 1, -1, 0],
              [0, -1, 1, 2, 1],
              [1, 2, 1, 1, 1]], dtype=float)
M = len(X)
N = len(X[0])

# 卷积核
W = np.array([[1, 0, 0],
              [0, 0, 0],
              [0, 0, -1]], dtype=float)
U = len(W)
V = len(W[0])
# 翻转
W_rot180 = rot180(W)

# 卷积
Y = np.zeros_like(X)
for i in range(U - 1, M):
    for j in range(V - 1, N):
        for u in range(U):
            for v in range(V):
                Y[i][j] += W_rot180[u, v] * X[i - u, j - v]
Y = Y[U - 1:, V - 1:]
print(Y)

# 互相关
Y = np.zeros_like(W)
for i in range(U):
    for j in range(V):
        for u in range(U):
            for v in range(V):
                Y[i][j] += W[u, v] * X[i + u, j + v]
print(Y)

'''
[[ 0.  2.  1.]
 [-2. -2. -4.]
 [ 1.  0.  0.]]
[[ 0.  2.  1.]
 [-2. -2. -4.]
 [ 1.  0.  0.]]
'''
