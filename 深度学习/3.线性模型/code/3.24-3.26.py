import numpy as np
import math


def sigma(x):  # Logistic激活函数
    return 1 / (1 + math.exp(-x))


N = 3  # 训练集数量
x_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # 特征向量
x_train = np.concatenate((x_train, np.ones((N, 1))), axis=1)  # 增广特征向量
y_train = np.array([1, 0, 1])  # 类别标签
w = np.array([1, 2, 3, 1])  # 增广权重向量

y_predict = np.zeros(N)  # 类别标签1的后验概率
for n in range(N):
    y_predict[n] = sigma(np.vdot(w, x_train[n]))

sum = np.zeros_like(x_train[0])  # 损失的和
for n in range(N):
    sum += x_train[n] * (y_train[n] - y_predict[n])

R_w_diff = -sum / N  # 损失函数关于权重向量的偏导数

print(R_w_diff)
'''
[0.12801409 0.15800012 0.18798614 0.29986022]
'''
