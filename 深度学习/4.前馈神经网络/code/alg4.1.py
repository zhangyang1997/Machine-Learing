import numpy as np


def sigma(x):
    '''
    激活函数sigma(x)
    '''
    return 1 / (1 + np.exp(-x))


def sigma_diff(x):
    '''
    激活函数sigma(x)的导函数
    '''
    return sigma(x) * sigma(1 - x)


def softmax(x):
    '''
    激活函数softmax(x)
    '''
    return np.exp(x) / (np.vdot(np.ones(C), np.exp(x)))


'''数据集'''
# 类别数量
C = 4
# 样本维数
D = 2

'''训练集'''
# 训练集特征矩阵
X_train = np.array([[1, 2],
                    [-3, 4],
                    [-1, -2],
                    [3, -4]])
# 训练集样本数量
N = len(X_train)
# 训练集标签
y_train = np.array([0, 1, 2, 3])
# one-hot向量
y_train_v = np.zeros((N, C))
for n in range(N):
    y_train_v[n, y_train[n]] = 1


'''验证集'''
# 验证集特征矩阵
X_valid = np.array([[12, 14],
                    [11, -12],
                    [-13, -14],
                    [-11, 12]])
# 验证集样本数量
N_valid = len(X_valid)
# 验证集标签
y_valid = np.array([0, 3, 2, 1])
# one-hot向量
y_valid_v = np.zeros((N_valid, C))
for n in range(N_valid):
    y_valid_v[n, y_valid[n]] = 1


'''超参数'''
# 学习率
alpha = 0.1
# 正则化系数
lambda_ = 0.01
# 训练轮数
T = 100

# 网络层数(不包括输入层)
L = 2
# 第0层神经元个数(特征向量的维数)
M_0 = D
# 第1层神经元个数
M_1 = 4
# 第2层神经元个数(类别数量)
M_2 = C

# 第0层到第1层的权重矩阵
W_1 = np.random.rand(M_1, M_0)
# 第0层到第1层的偏置
b_1 = np.ones((M_1, 1))

# 第1层到第2层的权重矩阵
W_2 = np.random.rand(M_2, M_1)
# 第1层到第2层的偏置
b_2 = np.ones((M_2, 1))

'''训练过程'''
for t in range(T):
    # 随机重排列
    per = np.random.permutation(X_train.shape[0])
    X_train_per = X_train[per, :]
    train_per = y_train_v[per]
    for n in range(N):
        '''1.前馈计算每一层的净输入z和激活值a'''
        # 第0层神经元的输出(即x)
        a_0 = X_train[n][:, np.newaxis]
        # 第1层神经元的净输入
        z_1 = np.dot(W_1, a_0) + b_1
        # 第1层神经元的输出
        a_1 = sigma(z_1)
        # 第2层神经元的净输入
        z_2 = np.dot(W_2, a_1) + b_2
        # 第2层神经元的输出(即y)
        a_2 = softmax(z_2)
        # 每个类的预测条件概率
        y_predict_v = a_2

        '''2.反向传播计算每一层的误差delta'''
        # 第2层神经元的误差项(交叉熵损失loss关于z的导数)
        delta_2 = y_predict_v - y_train_v[n][:, np.newaxis]
        # 第1层神经元的误差项
        delta_1 = sigma_diff(z_1) * (np.dot(W_2.T, delta_2))

        '''3.计算每一层参数的导数'''
        # loss关于第2层权重的梯度
        loss_diff_W_2 = np.dot(delta_2, a_1.T)
        # loss关于第2层偏置的梯度
        loss_diff_b_2 = delta_2

        # loss关于第1层权重的梯度
        loss_diff_W_1 = np.dot(delta_1, a_0.T)
        # loss关于第1层偏置的梯度
        loss_diff_b_1 = delta_1

        '''4.更新参数'''
        W_1 -= alpha * (loss_diff_W_1 + lambda_ * W_1)
        b_1 -= alpha * delta_1
        W_2 -= alpha * (loss_diff_W_2 + lambda_ * W_2)
        b_2 -= alpha * delta_2

    '''验证集错误率'''
    count = 0
    y_predict = np.zeros(N_valid)
    for n in range(N_valid):
        # 第0层神经元的输出(即x)
        a_0 = X_valid[n][:, np.newaxis]
        # 第1层神经元的净输入
        z_1 = np.dot(W_1, a_0) + b_1
        # 第1层神经元的输出
        a_1 = sigma(z_1)
        # 第2层神经元的净输入
        z_2 = np.dot(W_2, a_1) + b_2
        # 第2层神经元的输出(即y)
        a_2 = softmax(z_2)
        # 每个类的预测条件概率
        y_predict_v = a_2
        # 预测的类别
        y_predict[n] = np.argmax(y_predict_v)
        # 预测错误计数
        if y_valid[n] != y_predict[n]:
            count += 1
    print(y_predict)
    # 如果预测错误数量不再下降，中断
    if count == 0:
        break
print(t)
'''
[2. 1. 3. 2.]
[2. 1. 3. 2.]
[1. 0. 3. 1.]
[1. 3. 3. 1.]
[1. 3. 3. 1.]
[1. 3. 3. 1.]
[1. 3. 3. 1.]
[1. 3. 3. 1.]
[1. 3. 3. 1.]
[1. 3. 2. 1.]
[0. 3. 2. 1.]
10
'''
