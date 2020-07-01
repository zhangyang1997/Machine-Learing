import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def gelu(x):  # 近似GELU函数
    return x * sigma(1.702 * x)


gelu_x = 0
x = 1

gelu_x = gelu(x)

print(gelu_x)
'''
0.8457957659328212
'''
