import numpy as np
import math

gamma = [0.001, 0.002, 0.003]


def p_relu(i, x):  # 带参数的Relu函数
    if x > 0:
        return x
    else:
        return gamma[i] * x


p_relu_i_x = 0  # 输出
x = 1  # 输入
i = 1  # 神经元的序号


p_relu_i_x = p_relu(i, x)  # 公式4.21-4.22

print(p_relu_i_x)

'''
1
'''
