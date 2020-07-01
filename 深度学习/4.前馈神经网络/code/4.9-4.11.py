import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def sigma_fd(x):  # Logistic函数的一阶导函数
    return sigma(x)*(1-sigma(x))


def g_l(x):  # Logistic函数的一阶泰勒展开函数
    return sigma(0) + x * sigma_fd(0)


def hard_logistic(x):  # 分段函数hard_logistic近似logistic函数
    if g_l(x) >= 1:
        return 1
    elif g_l(x) <= 0:
        return 0
    else:
        return g_l(x)


hard_logistic_x = 0  # 输出
x = 1  # 输入

hard_logistic_x = hard_logistic(x)  # 公式4.9-4.11

print(hard_logistic_x)

'''
0.75
'''
