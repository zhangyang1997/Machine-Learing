import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N：batcH size(样本数量)
# D_in：输入层神经元数量(特征维数)
# H：隐藏层神经元数量
# D_out：输出层神经元数量
N, D_in, H, D_out = 64, 1000, 100, 10

# 输入特征矩阵
x = torch.randn(N, D_in, device=device, dtype=dtype)
# 输出结果矩阵
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 第0层到第1层的权重矩阵
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# 第1层到第2层的权重矩阵
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

# 学习率
learning_rate = 1e-6

# 迭代次数
for t in range(500):
    '''前向计算'''
    # 输入层的输出/隐藏层的输入
    h = x.mm(w1)
    # 隐藏层的输出/输出层的输入
    h_relu = h.clamp(min=0)
    # 输出层的输出
    y_pred = h_relu.mm(w2)

    '''平方误差损失函数'''
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    '''反向传播计算损失关于权重的梯度'''
    # loss关于Y_pred的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    # loss关于W2的梯度
    grad_w2 = h_relu.t().mm(grad_y_pred)
    # loss关于H_relu的梯度
    grad_h_relu = grad_y_pred.mm(w2.t())
    # loss关于H的梯度
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    # loss关于W1的梯度
    grad_w1 = x.t().mm(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
