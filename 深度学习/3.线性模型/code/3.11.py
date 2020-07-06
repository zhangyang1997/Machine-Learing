import numpy as np
import math

w = np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]])  # 权重向量
b = 1  # 偏置


def f(x, w_c):
    return np.vdot(w_c, x) + b


C = 3  # 类别数
f_c = np.zeros(C)  # 输出
x = np.array([1, 2, 3])  # 输入

for c in range(C):
    f_c[c] = f(x, w[c])

y = 0  # 预测类别
for c in range(1, C):
    if f_c[c] > f_c[y]:
        y = c

print(y)
'''
1
'''