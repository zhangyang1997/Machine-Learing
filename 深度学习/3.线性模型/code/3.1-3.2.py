import numpy as np
import math

w = np.asarray([1, 2, 3])  # 权重向量
b = 1  # 偏置


def f(x):
    return np.vdot(w, x) + b


x = [1, 2, 3]  # 输入
f_x = f(x)  # 输出

print(f_x)
'''
15
'''
