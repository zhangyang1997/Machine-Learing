import numpy as np
import math

w = np.array([1, 2, 3, 1])  # 增广权重向量


def f(x):  # 线性函数
    return np.vdot(w, x)


def sigma(x):  # Logistic激活函数
    return 1 / (1 + math.exp(-x))


x = np.array([1, 2, 3])  # 特征向量
x = np.concatenate((x, [1]), axis=0)  # 增广特征向量

p_1_x = sigma(f(x))  # 类别标签1的后验概率

print(p_1_x)
'''
0.999999694097773
'''
