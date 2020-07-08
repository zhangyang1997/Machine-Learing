import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def sigma_fd(x):  # Logistic函数的一阶导函数
    return sigma(x)*(1-sigma(x))


def g_l(x):  # Logistic函数的一阶泰勒展开函数
    return sigma(0) + x * sigma_fd(0)


g_l_x = 0  # 输出
x = 1  # 输入

g_l_x = g_l(x)  # 公式4.7-4.8

print(g_l_x)

'''
0.75
'''