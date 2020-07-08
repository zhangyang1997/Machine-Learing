import numpy as np
import math


def f(x):
    w = np.asarray([1, 2, 3])  # 权重向量
    b = 1  # 偏置
    return np.vdot(w, x) + b


def sgn(x):  # 符号函数
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


y = 0  # 输出
x = [1, 2, 3]  # 输入

y = sgn(f(x))  # 公式3.3-3.5

print(y)
'''
15
'''
