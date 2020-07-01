import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def tanh_fd(x):  # Tanh函数的一阶导函数
    return 1 - tanh(x) ** 2


def g_t(x):  # Tanh函数在0附近的一阶泰勒展开函数
    return tanh(0) + x * tanh_fd(0)


def hard_tanh(x):  # 分段函数hard_tanh近似tanh函数
    if g_t(x) >= 1:
        return 1
    elif g_t(x) <= -1:
        return - 1
    else:
        return g_t(x)


hard_tanh_x = 0  # 输出
x = 1  # 输入

hard_tanh_x = hard_tanh(x)  # 公式4.14-4.15

print(hard_tanh_x)

'''
1
'''
