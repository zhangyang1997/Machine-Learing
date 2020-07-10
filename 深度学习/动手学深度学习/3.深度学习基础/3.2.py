import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import IPython.display as display

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


def use_svg_display():
    '''用矢量图显示'''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    '''设置图的尺寸'''
    # 用矢量图显示
    use_svg_display()
    # 设置尺寸
    plt.rcParams['figure.figsize'] = figsize


# 设置图的尺寸
set_figsize()
# 绘制散点图
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# 显示
# plt.show()

'''2.读取数据'''
def data_iter(batch_size, features, labels):
    '''返回batch_size个随机样本的特征和标签'''
    # 样本数量
    num_examples = len(features)
    # 样本索引
    indices = list(range(num_examples))
    # 样本随机重排序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # batch_size大小的样本索引存储到j，最后一次可能不足一个batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        # 返回生成器对象
        yield features.index_select(0, j), labels.index_select(0, j)


# batch_size大小
batch_size = 10

'''3.初始化模型参数'''
# 权重参数，正态分布初始化(均值为0，标准差为0.01)，形状[2,1]
w = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, 1)), dtype=torch.float32)
# 偏置初始化为0
b = torch.zeros(1, dtype=torch.float32)
# 设置w和b是需要求梯度的参数
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

'''4.定义模型(线性函数)'''
def linreg(X, w, b):
    return torch.mm(X, w) + b


'''5.定义损失函数(均方误差损失)'''
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.shape)) ** 2 / 2


'''6.定义优化算法(更新权重参数)'''
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


'''7.训练模型'''
# 学习率
lr = 0.03
# 训练轮数
num_epochs = 3
# 网络结构
net = linreg
# 损失函数
loss = squared_loss
# 训练过程
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 当前样本均方误差损失结果
        l = loss(net(X, w, b), y).sum()
        # 反向传播计算损失关于权重的梯度
        l.backward()
        # 使用小批量随机梯度下降迭代权重参数
        sgd([w, b], lr, batch_size)
        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    # 训练集整体的损失
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
