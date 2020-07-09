import numpy as np
import math


def sigma(x):  # Logistic激活函数
    return 1 / (1 + math.exp(-x))


x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # 特征向量
x = np.concatenate((x, np.ones((3, 1))), axis=1)  # 增广特征向量
w = np.array([1, 2, 3, 1])  # 增广权重向量

N = 3  # 样本数量
y = np.zeros(N)  # 类别标签1的后验概率
for n in range(N):
    y[n] = sigma(np.vdot(w, x[n]))

print(y)
'''
[0.9168273  0.98522597 0.99752738]
'''
