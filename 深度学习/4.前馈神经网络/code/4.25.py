import numpy as np
import math


def softplus(x):  # Softplus函数
    return math.log(1 + math.exp(x))


softplus_x = 0  # 输出
x = 1  # 输入

softplus_x = softplus(x)  # 公式4.25

print(softplus_x)

'''
1.3132616875182228
'''