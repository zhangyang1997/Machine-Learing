import numpy as np


def f(x):
    '''
    神经元的激活函数sigma(x)
    '''
    return 1 / (1 + np.exp(-x))


# 第0层神经元个数
M_0 = 3
# 第1层神经元个数
M_1 = 4
# 第2层神经元个数
M_2 = 1

# 第0层神经元的输出(即x)
a_0 = np.array([[1, 2, 3]]).T

# 第0层到第1层的权重矩阵
W_1 = np.ones((M_1, M_0), dtype=float)
# 第0层到第1层的偏置
b_1 = np.ones((M_1, 1))
# 第1层神经元的输出
a_1 = f(np.dot(W_1, a_0) + b_1)

# 第1层到第2层的权重矩阵
W_2 = np.ones((M_2, M_1), dtype=float)
# 第1层到第2层的偏置
b_2 = np.ones((M_2, 1))
# 第2层神经元的输出(即y)
a_2 = f(np.dot(W_2, a_1) + b_2)
# 类别y=1的条件概率
p = a_2
print(p)

'''
[[0.99328288]]
'''
