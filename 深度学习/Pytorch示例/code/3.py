import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N：batcH size(样本数量)
# D_in：输入层神经元数量(特征维数)
# H：隐藏层神经元数量
# D_out：输出层神经元数量
N, D_in, H, D_out = 64, 1000, 100, 10

'''requires_grad=False/True表示不需要/需要计算梯度'''

# 输入特征矩阵
x = torch.randn(N, D_in, device=device, dtype=dtype)
# 输出结果矩阵
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 第0层到第1层的权重矩阵
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# 第1层到第2层的权重矩阵
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# 学习率
learning_rate = 1e-6
for t in range(500):
    '''前向计算，不需要保留中间值'''
    # 输出层的输出
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    '''平方误差损失函数'''
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    '''反向传播计算损失关于权重的梯度'''
    loss.backward()

    '''更新权重'''
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 更新权重后将梯度置零
        w1.grad.zero_()
        w2.grad.zero_()