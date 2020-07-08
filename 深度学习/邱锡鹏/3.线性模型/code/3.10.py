import numpy as np
import math

w = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 0]])  # 权重向量
b = 1  # 偏置


def f(x, w_c):
    return np.vdot(w_c, x) + b


C = 3
f_c = np.zeros(C)
x = np.array([1, 2, 3])

for c in range(C):
    f_c[c] = f(x, w[c])

print(f_c)
'''
[ 7. -5.  1.]
'''
