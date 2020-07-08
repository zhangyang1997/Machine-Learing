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
# 样本维数
D = 3
# 训练集样本数量
N = 4

# 第0层神经元个数(特征向量的维数)
M_0 = D
# 第1层神经元个数
M_1 = 4
# 第2层神经元个数(类别数量)
M_2 = C

# 训练集特征矩阵
x_train = np.random.rand(N, D)

# 训练集标签
y_train = np.array([3, 0, 5, 1])
# one-hot向量
y_train_v = np.zeros((N, C))
for n in range(N):
    y_train_v[n, y_train[n]] = 1


# 第0层到第1层的权重矩阵
W_1 = np.random.rand(M_1, M_0)
# 第0层到第1层的偏置
b_1 = np.ones((M_1, 1))

# 第1层到第2层的权重矩阵
W_2 = np.random.rand(M_2, M_1)
# 第1层到第2层的偏置
b_2 = np.ones((M_2, 1))

sum = 0
for n in range(N):
    # 第0层神经元的输出(即x)
    a_0 = x_train[n, np.newaxis].T

    # 第1层神经元的输出
    a_1 = f(np.dot(W_1, a_0) + b_1)

    # 第2层神经元的输出(即y)
    a_2 = softmax(np.dot(W_2, a_1) + b_2)

    # 每个类的条件概率
    y_predict_v = a_2

    # 交叉熵损失
    loss = -np.vdot(y_train_v[n], np.log(y_predict_v))
    sum += loss

# 正则化参数
lambda_ = 0.1
# 正则化项
W_F = np.linalg.norm(W_1)**2 + np.linalg.norm(W_2)**2
# 结构风险
R_W = sum / N + lambda_ * W_F
print(R_W)
'''
3.6696912312968877
'''
