import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def gelu(x):  # 近似GELU函数
    return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


gelu_x = 0  # 输出
x = 1  # 输入

gelu_x = gelu(x)  # 公式4.28

print(gelu_x)
'''
0.8411919906082768
'''
