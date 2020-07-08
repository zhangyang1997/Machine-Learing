import numpy as np
import math


def leaky_relu(x):  # 带泄露的Relu函数
    gamma = 0.001
    if x > 0:
        return x
    else:
        return gamma * x


leaky_relu_x = 0  # 输出
x = 1  # 输入

leaky_relu_x = leaky_relu(x)  # 公式4.18-4.19

print(leaky_relu_x)

'''
1
'''