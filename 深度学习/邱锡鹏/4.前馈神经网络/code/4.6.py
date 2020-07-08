import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def tanh(x):  # Tanh函数
    return 2 * sigma(2 * x) - 1


tanh_x = 0  # 输出
x = 1  # 输入

tanh_x = tanh(x)  # 公式4.6

print(tanh_x)

'''
0.7615941559557646
'''