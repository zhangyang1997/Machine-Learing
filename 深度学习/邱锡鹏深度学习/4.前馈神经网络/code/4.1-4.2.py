import numpy as np

z = 0  # 净输入
w = np.asarray([1, 2, 3])  # 3维权重向量
x = np.asarray([1, 2, 3])  # 输入向量
b = 1  # 偏置

z = np.vdot(w, x) + b  # 公式4.1，4.2

print(z)

'''
15
'''
