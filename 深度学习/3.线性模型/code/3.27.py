import numpy as np
import math


def sigma(x):  # Logistic激活函数
    return 1 / (1 + math.exp(-x))


alpha = 0.5  # 学习率
N = 3  # 训练集数量
x_train = np.array(
    [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [0.7, 0.8, 0.9]])  # 特征向量
x_train = np.concatenate((x_train, np.ones((N, 1))), axis=1)  # 增广特征向量
y_train = np.array([1, 0, 1])  # 类别标签
w = np.array([-0.1, 0.2, -0.3, 0.5], dtype=float)  # 增广权重向量


T = 10  # 训练轮数
for t in range(T):
    y_predict = np.zeros(N)  # 类别标签1的后验概率
    for n in range(N):
        y_predict[n] = sigma(np.vdot(w, x_train[n]))

    sum = np.zeros_like(x_train[0])  # 损失的和
    for n in range(N):
        sum += x_train[n] * (y_train[n] - y_predict[n])

    w += alpha * sum / N  # 更新权重
    print(y_predict)
'''
[0.60825903 0.65021855 0.57932425]
[0.63318576 0.61233977 0.65653881]
[0.65295863 0.57578564 0.71714826]
[0.66895553 0.54105732 0.76437727]
[0.68221053 0.50846591 0.80137089]
[0.69346319 0.47816452 0.83065886]
[0.70323397 0.45018472 0.85414002]
[0.7118888  0.42447157 0.8732071 ]
[0.71968664 0.40091377 0.88887866]
[0.72681262 0.37936736 0.90190439]
'''
