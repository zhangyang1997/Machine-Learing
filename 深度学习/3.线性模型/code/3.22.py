import numpy as np
import math


def sigma(x):  # Logistic激活函数
    return 1 / (1 + math.exp(-x))


x_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # 特征向量
x_train = np.concatenate((x_train, np.ones((3, 1))), axis=1)  # 增广特征向量
y_train = np.array([1, 0, 1])  # 类别标签
w = np.array([1, 2, 3, 1])  # 增广权重向量

N = 3  # 训练集数量
y_predict = np.zeros(N)  # 类别标签1的后验概率
for n in range(N):
    y_predict[n] = sigma(np.vdot(w, x_train[n]))

sum = 0
for n in range(N):
    sum += y_train[n] * math.log(y_predict[n]) + \
        (1 - y_train[n]) * math.log(1 - y_predict[n])
R_w = -sum / N  # 交叉熵损失函数

print(R_w)
'''
1.4347320306545324
'''
