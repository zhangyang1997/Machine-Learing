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


'''2.计算损失'''
# N个样本类别标签的后验概率向量组成的矩阵
y_predict = np.zeros((C, N))
for n in range(N):
    y_predict[:, n] = softmax(np.dot(W.T, x_train[:, n]), C)

# 损失函数关于去增广权重矩阵的梯度
sum = 0
for n in range(N):
    sum += np.vdot(y_train_v[:, n], np.log(y_predict[:, n]))
R_W = -sum / N

print(R_W)
'''
2.1429316284998996
'''
