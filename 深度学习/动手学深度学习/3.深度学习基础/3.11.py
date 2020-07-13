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

'''1.生成数据集'''
# 训练集大小，测试集大小，权重参数，偏置参数
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# 随机创建特征向量
features = torch.randn((n_train + n_test, 1))
# 多项式(x,x^2,x^3)
poly_features = torch.cat(
    (features, torch.pow(features, 2), torch.pow(features, 3)), 1)
# 标签
labels = (true_w[0] * poly_features[:, 0] + true_w[1] *
          poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b)
# 带噪声的标签
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float)

'''2.定义、训练和测试模型'''
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


num_epochs, loss = 100, torch.nn.MSELoss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    # 网络结构
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 批大小
    batch_size = min(10, train_labels.shape[0])
    # 数据集
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    # 训练集迭代器
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 训练集损失，测试集损失
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 损失
            l = loss(net(X), y.view(-1, 1))
            # 参数梯度清零
            optimizer.zero_grad()
            # 反向传播计算梯度
            l.backward()
            # 梯度下降更新权重
            optimizer.step()
        # 训练集标签
        train_labels = train_labels.view(-1, 1)
        # 测试集标签
        test_labels = test_labels.view(-1, 1)
        # 训练集损失
        train_ls.append(loss(net(train_features), train_labels).item())
        # 测试集损失
        test_ls.append(loss(net(test_features), test_labels).item())
    print('最后一轮: 训练集损失', train_ls[-1], '测试集损失', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('权重:', net.weight.data,
          '\n偏置:', net.bias.data)
    print()


fit_and_plot(poly_features[:n_train, :],
             poly_features[n_train:, :], labels[:n_train], labels[n_train:])

fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])

fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])
