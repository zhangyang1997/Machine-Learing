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
# 特征向量组成的矩阵shape=(d,N)
x_train = np.array([[1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9]], dtype=float)
# 增广特征向量组成的矩阵shape=(d+1,N)
x_train = np.concatenate((x_train, np.ones((1, N))), axis=0)
# 类别标签组成的向量shape=(3,1)
y_train = np.array([[2],
                    [0],
                    [1]])
# 标签对应的one-hot向量组成的标签矩阵shape=(C,N)
y_train_v = np.zeros((C, N))
for n in range(N):
    y_train_v[y_train[n], n] = 1
# 增广权重向量组成的矩阵shape=(d+1,C)
W = np.array([[-1, 1, 1],
              [1, -1, 1],
              [1, 1, -1],
              [1, 1, 1]], dtype=float)


'''2.更新权重'''
T = 1000  # 迭代次数
for t in range(T):

    # N个样本类别标签的后验概率向量组成的矩阵
    y_predict = np.zeros((C, N))
    for n in range(N):
        y_predict[:, n] = softmax(np.dot(W.T, x_train[:, n]), C)

    # 损失函数关于去增广权重矩阵的梯度
    sum = np.zeros((N+1, C))
    for n in range(N):
        sum += np.dot(x_train[:, n, np.newaxis],
                      y_train_v[np.newaxis, :, n, ] - y_predict[np.newaxis, :, n])
    R_W_diff = -sum / N

    # 更新权重
    W -= alpha * (R_W_diff)

print(W)
'''
[[-0.63754141  4.82952481 -3.1919834 ]
 [ 1.06073555 -0.37663243  0.31589689]
 [ 0.7590125  -1.58278968  1.82377717]
 [ 0.69827696 -2.20615724  4.50788028]]
'''

for n in range(N):
    print(np.argmax(y_train_v[:, n]), np.argmax(y_predict[:, n]))
'''
2 2
0 0
1 1
'''
