import numpy as np


def f(x):
    '''
    激活函数sigma(x)
    '''
    return 1 / (1 + np.exp(-x))


def softmax(x):
    '''
    激活函数softmax(x)
    '''
    return np.exp(x) / (np.vdot(np.ones(C), np.exp(x)))


# 类别数量
C = 10
# 第0层神经元个数
M_0 = 3
# 第1层神经元个数
M_1 = 4
# 第2层神经元个数
M_2 = C

# 第0层神经元的输出(即x)
a_0 = np.array([[1, 2, 3]]).T

# 第0层到第1层的权重矩阵
W_1 = np.random.rand(M_1, M_0)
# 第0层到第1层的偏置
b_1 = np.ones((M_1, 1))
# 第1层神经元的输出
a_1 = f(np.dot(W_1, a_0) + b_1)

# 第1层到第2层的权重矩阵
W_2 = np.random.rand(M_2, M_1)
# 第1层到第2层的偏置
b_2 = np.ones((M_2, 1))
# 第2层神经元的输出(即y)
a_2 = softmax(np.dot(W_2, a_1) + b_2)
# 每个类的条件概率
y_predict_v = a_2

print(y_predict_v)
'''
[[0.04621109]
 [0.12578412]
 [0.19063479]
 [0.02744378]
 [0.0937828 ]
 [0.03752134]
 [0.08774394]
 [0.17376121]
 [0.09420282]
 [0.12291411]]
'''
