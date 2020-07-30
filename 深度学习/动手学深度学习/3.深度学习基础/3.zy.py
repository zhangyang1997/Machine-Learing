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
import torchsummary


# 1.获取和读取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST/', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 2.定义网络结构
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
num_inputs = 784
num_outputs = 10
num_hiddens = 256
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)
print(torchsummary.summary(net, (1,784)))

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# 3.定义损失函数(输出层softmax+平均交叉熵损失)
loss = torch.nn.CrossEntropyLoss()

# 4.定义优化算法(更新权重参数)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 5.定义分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += len(y)
    return acc_sum / n

# 6.训练模型
num_epochs = 5
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None, lr=None, optimizer=None):
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

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)



'''
轮数 1, 训练集平均损失 0.0032, 训练集分类准确率 0.696, 测试集分类准确率 0.716
轮数 2, 训练集平均损失 0.0019, 训练集分类准确率 0.818, 测试集分类准确率 0.800
轮数 3, 训练集平均损失 0.0016, 训练集分类准确率 0.844, 测试集分类准确率 0.841
轮数 4, 训练集平均损失 0.0016, 训练集分类准确率 0.854, 测试集分类准确率 0.837
轮数 5, 训练集平均损失 0.0015, 训练集分类准确率 0.863, 测试集分类准确率 0.852
'''