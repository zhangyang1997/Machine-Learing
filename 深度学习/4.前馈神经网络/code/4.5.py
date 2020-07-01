import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


tanh_x = 0  # 输出
x = 1  # 输入

tanh_x = tanh(x)  # 公式4.5

print(tanh_x)

'''
0.7615941559557649
'''