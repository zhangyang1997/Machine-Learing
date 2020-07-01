import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def swish(x):  # swish函数
    # beta = 0  #线性函数
    beta = 1  # x>0线性，x<0近似饱和
    # beta = 100  #近似relu函数
    return x * sigma(beta * x)


swish_x = 0  # 输出
x = 1  # 输入

swish_x = swish(x)  # 公式4.26

print(swish_x)
'''
0.7310585786300049
'''