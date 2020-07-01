import numpy as np
import math


def elu(x):  # ELU函数，指数线性单元函数
    gamma = 0.001
    if x > 0:
        return x
    else:
        return gamma*(math.exp(x)-1)


elu_x = 0  # 输出
x = 1  # 输入

elu_x = elu(x)  # 公式4.23-4.24

print(elu_x)

'''
1
'''
