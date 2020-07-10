import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

'''1.生成数据集'''
# 特征维数
num_inputs = 2
# 样本数量
num_examples = 1000
# 真实权重
true_w = [2, -3.4]
# 真实偏置
true_b = 4.2
# 样本特征矩阵，形状[1000，2]
features = torch.tensor(np.random.normal(
    0, 1, (num_examples, num_inputs)), dtype=torch.float)
# 样本真实标签，形状[1000]
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 样本真实标签加入噪声
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float)

'''2.读取数据'''
# batch_size大小
batch_size = 10
# 特征矩阵和标签向量组成数据集对象
dataset = Data.TensorDataset(features, labels)
# 返回数据集生成器，并设置参数
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

'''3.定义模型'''
# 网络结构
net = nn.Sequential(nn.Linear(num_inputs, 1))
print(net)

'''4.初始化模型参数'''
# 权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布
init.normal_(net[0].weight, mean=0, std=0.01)
# 偏置为0，等价net[0].bias.data.fill_(0)
init.constant_(net[0].bias, val=0)

'''5.定义损失函数(均方误差损失)'''
# 均方误差损失函数
loss = nn.MSELoss()

'''6.定义优化算法(更新权重参数)'''
optimizer = optim.SGD(net.parameters(), lr=0.03)

'''7.训练模型'''
# 训练轮数
num_epochs = 3
# 训练过程
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        # 预测标签
        output = net(X)
        # 当前样本均方误差损失结果
        l = loss(output, y.view(output.shape))
        # 反向传播计算损失关于权重的梯度
        l.backward()
        # 使用小批量随机梯度下降迭代权重参数
        optimizer.step()
        # 梯度清零，等价于net.zero_grad()
        optimizer.zero_grad()
    print('epoch %d, loss: %f' % (epoch, l.item()))

print()

f = net[0]
print(true_w, f.weight)
print(true_b, f.bias)
