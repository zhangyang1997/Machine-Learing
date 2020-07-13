
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

'''丢弃法'''
def dropout(X, drop_prob):
    # 转float
    X = X.float()
    # 丢弃概率
    assert 0 <= drop_prob <= 1
    # 保留概率
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    # 保留的mask对应元素为1
    mask = (torch.rand(X.shape) < keep_prob).float()
    # 除保留元素其他元素为0
    return mask * X / keep_prob

'''返回数据集迭代器'''
def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST/', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter,test_iter


'''初始化参数'''
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)
# 参数
params = [W1, b1, W2, b2, W3, b3]

'''定义网络结构'''
# 丢弃概率
drop_prob1, drop_prob2 = 0.2, 0.5
def net(X, is_training=True):
    # 输入层的输入
    X = X.view(-1, num_inputs)
    # 隐藏层1的输出
    H1 = (torch.matmul(X, W1) + b1).relu()
    # 只在训练模型时使用丢弃法
    if is_training:
        # 在第一层全连接后添加丢弃层
        # 隐藏层1的输出后添加丢弃层
        H1 = dropout(H1, drop_prob1)
    # 隐藏层2的输出
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        # 在第二层全连接后添加丢弃层
        # 隐藏层2的输出后添加丢弃层
        H2 = dropout(H2, drop_prob2)
    # 返回输出层的输入
    return torch.matmul(H2, W3) + b3

'''分类准确率'''
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式
        else: # 自定义的模型
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
        n += y.shape[0]
    return acc_sum / n

'''训练和测试模型'''
num_epochs, lr, batch_size = 5, 0.5, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = load_data_fashion_mnist(batch_size)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    optimizer = torch.optim.SGD(params, lr)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('轮数 %d, 训练集平均损失 %.4f, 训练集分类准确率 %.3f, 测试集分类准确率 %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

'''
轮数 1, 训练集平均损失 0.0043, 训练集分类准确率 0.577, 测试集分类准确率 0.740
轮数 2, 训练集平均损失 0.0022, 训练集分类准确率 0.792, 测试集分类准确率 0.796
轮数 3, 训练集平均损失 0.0019, 训练集分类准确率 0.824, 测试集分类准确率 0.837
轮数 4, 训练集平均损失 0.0018, 训练集分类准确率 0.837, 测试集分类准确率 0.826
轮数 5, 训练集平均损失 0.0016, 训练集分类准确率 0.848, 测试集分类准确率 0.810
'''