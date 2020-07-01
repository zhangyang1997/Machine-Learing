import numpy as np


def f(x):  # 非线性函数f，假设为正弦函数
    return np.sin(x)


a = 0  # 活性值
z = 15  # 净输入

a = f(15)  # 公式4.3

print(a)

'''
0.6502878401571168
'''
