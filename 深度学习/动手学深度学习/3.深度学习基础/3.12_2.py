
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

'''画图'''
def use_svg_display():
    '''用矢量图显示'''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    '''设置图的尺寸'''
    # 用矢量图显示
    use_svg_display()
    # 设置尺寸
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    '''作图函数'''
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

'''1.生成数据集'''
# 训练集数量，测试集
n_train, n_test, num_inputs = 20, 100, 200
# 真实权重，偏置 
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
# 特征向量
features = torch.randn((n_train + n_test, num_inputs))
# 标签
labels = torch.matmul(features, true_w) + true_b
# 带噪声的标签
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 训练集，测试集特征向量
train_features, test_features = features[:n_train, :], features[n_train:, :]
# 训练集，测试集标签
train_labels, test_labels = labels[:n_train], labels[n_train:]

'''2.定义损失函数(均方误差损失)'''
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.shape)) ** 2 / 2

'''3.定义训练和测试'''
# 批大小，训练轮数，学习率
batch_size, num_epochs, lr = 1, 100, 0.003
# 数据集
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
# 训练集迭代器
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
# 损失函数
loss = squared_loss

def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一般是以weight结尾
    # 网络结构
    net = nn.Linear(num_inputs, 1)
    # 参数初始化
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    # 权重参数优化算法(weight_decay权重衰减)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    # 偏置优化算法(不对偏置衰减)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr) 

    # 训练集损失，测试集损失
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # bacth内的平均损失
            l = loss(net(X), y).mean()
            # 参数的梯度清零
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            # 反向传播计算梯度
            l.backward()
            # 梯度下降更新参数对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        # 训练集平均损失
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        # 测试集平均损失
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    # 画图
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())

# 过拟合
fit_and_plot_pytorch(0)
# 权重衰减防止过拟合
fit_and_plot_pytorch(3)