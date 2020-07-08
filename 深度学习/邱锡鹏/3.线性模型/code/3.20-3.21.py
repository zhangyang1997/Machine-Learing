import numpy as np
import math

x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # 特征向量
x = np.concatenate((x, np.ones((3, 1))), axis=1)  # 增广特征向量
y = np.array([1, 0, 1])  # 类别标签

N = 3  # 训练集数量
p_r_1 = np.zeros(N)  # 标签1的真实条件概率
p_r_0 = np.zeros(N)  # 标签0的真实条件概率
for n in range(N):
    p_r_1[n] = y[n]
    p_r_0[n] = 1 - y[n]

print(p_r_1)
print(p_r_0)
'''
[1. 0. 1.]
[0. 1. 0.]
'''
