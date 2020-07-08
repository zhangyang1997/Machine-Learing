import numpy as np
import math

w = np.array([1, 2, 3])  # 权重向量
b = 1  # 偏置


def f(x):
    return np.vdot(w, x) + b


def I(y, f):  # 指示函数
    if y * f > 0:
        return 1
    else:
        return - 1


x = np.array([1, 2, 3])  # 输入
y = 1  # 类别

L_01 = I(y, f(x))  # 输出

print(L_01)
'''
1
'''
