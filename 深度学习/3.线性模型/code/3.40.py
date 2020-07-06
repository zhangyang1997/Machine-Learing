import numpy as np


def softmax(W_x, C):  # Softmax函数
    return np.exp(W_x) / (np.vdot(np.ones(C), np.exp(W_x)))


'''1.数据预处理'''
# 训练样本数量
N = 3
# 类别数量
C = 3
# 学习率
alpha = 0.1
# 特征向量组成的矩阵
x_train = np.array([[1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9]], dtype=float)
# 增广特征向量组成的矩阵
x_train = np.concatenate((x_train, np.ones((1, N))), axis=0)
# 类别标签组成的向量
y_train = np.array([[0],
                    [1],
                    [2]])
# 标签对应的one-hot向量组成的标签矩阵
y_train_v = np.zeros((C, N))
for n in range(N):
    y_train_v[n, y_train[n]] = 1
# 增广权重向量组成的矩阵
W = np.array([[-1, 1, 1],
              [1, -1, 1],
              [1, 1, -1],
              [1, 1, 1]], dtype=float)


'''2.计算梯度'''
# N个样本类别标签的后验概率向量组成的矩阵
y_predict = np.zeros((C, N))
for n in range(N):
    y_predict[:, n] = softmax(np.dot(W.T, x_train[:, n]), C)

# 损失函数关于去增广权重矩阵的梯度
sum = np.zeros((N+1, C))
for n in range(N):
    sum += np.dot(np.array([x_train[:, n]]).T,
                  np.array([y_train_v[:, n] - y_predict[:, n]]))
R_W_diff = -sum / N

print(R_W_diff)
'''
[[ 3.13392    -0.86409162 -2.26982837]
 [ 3.66739999 -1.08011453 -2.58728547]
 [ 4.20087999 -1.29613743 -2.90474256]
 [ 0.53348    -0.21602291 -0.31745709]]
'''
