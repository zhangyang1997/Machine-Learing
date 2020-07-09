import numpy as np

# N：batcH size(样本数量)
# D_in：输入层神经元数量(特征维数)
# H：隐藏层神经元数量
# D_out：输出层神经元数量
N, D_in, D_h, D_out = 64, 1000, 100, 10

# 输入特征矩阵
X = np.random.randn(N, D_in)
# 输出结果矩阵
Y = np.random.randn(N, D_out)

# 第0层到第1层的权重矩阵
W1 = np.random.randn(D_in, D_h)
# 第1层到第2层的权重矩阵
W2 = np.random.randn(D_h, D_out)

# 学习率
learning_rate = 1e-6

# 迭代次数
for t in range(500):
    '''前向计算'''
    # 输入层的输出/隐藏层的净输入
    H = X.dot(W1)
    # 隐藏层的输出/输出层的净输入
    H_relu = np.maximum(H, 0)
    # 输出层的输出
    Y_pred = H_relu.dot(W2)

    '''平方误差损失函数'''
    loss = np.square(Y_pred - Y).sum()
    print(t, loss)

    '''反向传播计算损失关于权重的梯度'''
    # loss关于Y_pred的梯度
    grad_Y_pred = 2.0 * (Y_pred - Y)
    
    # loss关于W2的梯度
    grad_W2 = H_relu.T.dot(grad_Y_pred)

    # loss关于H_relu的梯度
    
    grad_H_relu = grad_Y_pred.dot(W2.T)
    
    # loss关于H的梯度
    grad_H = grad_H_relu.copy()
    grad_H[H < 0] = 0
    
    # loss关于W1的梯度
    grad_W1 = X.T.dot(grad_H)

    # 更新权重
    W1 -= learning_rate * grad_W1
    W2 -= learning_rate * grad_W2
