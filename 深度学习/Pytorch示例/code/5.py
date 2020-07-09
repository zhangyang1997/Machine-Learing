import torch

# N：batcH size(样本数量)
# D_in：输入层神经元数量(特征维数)
# H：隐藏层神经元数量
# D_out：输出层神经元数量
N, D_in, H, D_out = 64, 1000, 100, 10

# 输入特征矩阵
x = torch.randn(N, D_in)
# 输出结果矩阵
y = torch.randn(N, D_out)

# 网路结构
model = torch.nn.Sequential(
    # 输入层输出H
    torch.nn.Linear(D_in, H),
    # 隐藏层
    torch.nn.ReLU(),
    # 输出层输出D_out
    torch.nn.Linear(H, D_out),
)

# 均方误差损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')

# 学习率
learning_rate = 1e-4
for t in range(500):
    '''前向计算结果'''
    y_pred = model(x)

    '''平方误差损失'''
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    '''梯度置零'''
    model.zero_grad()

    '''反向传播计算损失关于权重参数的梯度'''
    loss.backward()

    '''更新权重参数'''
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
