import numpy as np
import math

w = np.array([1, 2, 3])  # 权重向量
b = 1  # 偏置


def f(x):
    return np.vdot(w, x) + b


N = 3  # 样本数量
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 输入
y = np.array([1, 1, -1])  # 类别


for n in range(N):
    if y[n] == 1:
        if f(x[n]) > 0:
            print("True")
        else:
            print("False")
    elif y[n] == -1:
        if f(x[n]) < 0:
            print("True")
        else:
            print("False")
    else:
        print("hyperplane")
'''
True
True
False
'''
