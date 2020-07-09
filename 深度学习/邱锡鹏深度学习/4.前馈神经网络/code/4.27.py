import numpy as np
import math
from scipy.stats import norm


def P(x):  # 高斯分布N(mu,sigma^2)的累积分布函数
    return norm.cdf(x, 0, 1)


def gelu(x):  # GELU函数，高斯误差线性单元
    return x * P(x)


gelu_x = 0  # 输出
x = 1  # 输入

gelu_x = gelu(x)  # 公式4.27

print(gelu_x)

'''
0.8413447460685429
'''
