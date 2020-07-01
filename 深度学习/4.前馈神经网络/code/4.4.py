import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


sigma_x = 0  # 输出
x = 1  # 输入

sigma_x = sigma(x)  # 公式4.4

print(sigma_x)

'''
0.7310585786300049
'''
