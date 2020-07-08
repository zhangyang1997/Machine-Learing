import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def tanh_fd(x):  # Tanh函数的一阶导函数
    return 1 - tanh(x) ** 2


def g_t(x):  # Tanh函数在0附近的一阶泰勒展开函数
    return tanh(0) + x * tanh_fd(0)


g_t_x = 0  # 输出
x = 1  # 输入

g_t_x = g_t(x)  # 公式4.12-4.13

print(g_t_x)

'''
1.0
'''