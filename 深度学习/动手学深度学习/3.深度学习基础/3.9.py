import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import IPython.display as display
import sys
import numpy as np
import torch.nn as nn
import collections
import torch.nn.init as init

'''1.获取和读取数据集'''
# 训练集
mnist_train = torchvision.datasets.FashionMNIST(
    root='./Datasets/FashionMNIST/', train=True, download=True, transform=transforms.ToTensor())
# 测试集
mnist_test = torchvision.datasets.FashionMNIST(
    root='./Datasets/FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())

batch_size = 256

if sys.platform.startswith('win'):
    # 0表示不用额外的进程来加速读取数据
    num_workers = 0
else:
    num_workers = 4
# 返回数据集迭代器
train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

'''2.定义模型参数'''
# 特征维数(输入层神经元数量)
num_inputs = 784
# 输出维数(输出层神经元数量)
num_outputs = 10
# 隐藏层神经元数量
num_hiddens = 256

# 第0层到第1层的权重 
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
# 第1层到第2层的权重
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
# 设置需要求梯度的参数
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

'''3.定义激活函数'''
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

'''4.定义模型'''
def net(X):
    # (batch_size,1,28,28)修改为(batch_size,784)
    X = X.view((-1, num_inputs))
    # 隐藏层的输出
    H = relu(torch.matmul(X, W1) + b1)
    # 输入层的输入
    return torch.matmul(H, W2) + b2

'''5.定义输出层softmax函数和交叉熵损失函数'''
loss = nn.CrossEntropyLoss()

'''6.计算分类准确率'''
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += len(y)
    return acc_sum / n

'''7.定义优化算法(更新权重参数)'''
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

'''8.训练模型'''
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    # 训练过程
    for epoch in range(num_epochs):
        # 损失，准确率
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 预测向量
            y_hat = net(X)
            # 损失
            l = loss(y_hat, y).sum()
            # 反向传播计算梯度前梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # 反向传播计算损失关于权重的梯度
            l.backward()
            # 使用小批量随机梯度下降迭代权重参数
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
            # 训练集损失和
            train_l_sum += l.item()
            # 训练集准确率的和
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            # 训练集数量
            n += y.shape[0]
        # 测试集准确率
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
              
# 训练轮数
num_epochs = 5
# 学习率
lr = 100
# 训练(网络结构，训练集迭代器，测试集迭代器，交叉熵损失函数，训练轮数，batch_size，权重参数，学习率)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
'''
epoch 1, loss 0.0030, train acc 0.712, test acc 0.794
epoch 2, loss 0.0019, train acc 0.822, test acc 0.783
epoch 3, loss 0.0017, train acc 0.844, test acc 0.830
epoch 4, loss 0.0015, train acc 0.856, test acc 0.838
epoch 5, loss 0.0014, train acc 0.865, test acc 0.856
'''