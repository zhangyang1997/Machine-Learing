import numpy as np

# 第0层神经元个数
M_0 = 3
# 第1层神经元个数
M_1 = 4

# 第0层到第1层的权重矩阵
W_1 = np.ones((M_1, M_0), dtype=float)
# 第0层神经元的输出(即x)
a_0 = np.array([[1, 2, 3]]).T
# 第0层到第1层的偏置
b_1 = np.ones((M_1, 1))
# 第1层神经元的净输入
z_1 = np.dot(W_1, a_0) + b_1
print(z_1)
'''
[[7.]
 [7.]
 [7.]
 [7.]]
'''
