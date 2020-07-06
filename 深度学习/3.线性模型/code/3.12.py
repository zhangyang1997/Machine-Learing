import numpy as np
import math

w = np.array([1, 1, 1])  # 权重向量
b = 1  # 偏置


def f(x):  # 线性函数
    return np.vdot(w, x) + b


def g(x):  # 激活函数(以Logistic函数为例)
    return 1 / (1 + math.exp(-x))


x = np.array([1, 2, 3])  # 输入
p_x = g(f(x))  # 类别标签的后验概率

print(p_x)
'''
0.9990889488055994
'''
