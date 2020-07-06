import numpy as np

N = 1  # 训练集数量
C = 3  # 类别数量
x_train = np.array([1, 2, 3])  # 特征向量
x_train = np.concatenate((x_train, np.ones(1)), axis=0)  # 增广特征向量
y_train = np.array(2)  # 类别标签
W = np.array([[-1, 1, -1, 1], [-1, 1, 1, 1],
              [1, 1, 1, 1]], dtype=float)  # 增广权重向量组成的矩阵


def softmax(W_x):  # Softmax函数
    return np.exp(W_x) / (np.vdot(np.ones(C), np.exp(W_x)))


y_predict = softmax(np.dot(W, x_train.T))  # 所有类别的预测条件概率组成的向量

print(y_predict)
'''
[2.95387223e-04 1.19167711e-01 8.80536902e-01]
'''
