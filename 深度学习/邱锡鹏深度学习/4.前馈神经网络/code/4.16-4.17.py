import numpy as np
import math


def relu(x):  # ReLU函数,修正线性单元函数
    if x >= 0:
        return x
    else:
        return 0


relu_x = 0  # 输出
x = 1  # 输入

relu_x = relu(x)  # 公式4.16-4.17

print(relu_x)

'''
1
'''
