import numpy as np
import math

K = 3  # 权重向量的个数
z = np.zeros(K)  # K个净输入组成的向量
w = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # K个权重向量组成的矩阵
x = np.asarray([1, 2, 3])  # 输入向量
b = np.asarray([1, 2, 3])  # K个偏置组成的向量

for k in range(K):
    z[k] = np.vdot(w[k], x) + b[k]  # 公式4.30

print(z)

'''
[15. 16. 17.]
'''
