import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import IPython.display as display
import sys
import numpy as np


def use_svg_display():
    '''用矢量图显示'''
    display.set_matplotlib_formats('svg')


'''1.获取数据集'''
# 训练集
mnist_train = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
# 测试集
mnist_test = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
print()

# 样本0，标签
feature, label = mnist_train[0]
# 通道数 x 高度 x 宽度
print(feature.shape, label)
print()

# 数值标签转文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 显示图像(图像，标签)
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 显示训练集前10个样本的图像和文本标签
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

'''2.读取小批量数据'''
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

'''3.初始化模型参数'''
# 特征维数
num_inputs = 784
# 输出维数
num_outputs = 10
# 权重
W = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
# 偏置
b = torch.zeros(num_outputs, dtype=torch.float)
# 需要求W和b的梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

'''4.实现softmax运算'''
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # 广播机制
    return X_exp / partition


'''5.定义模型'''
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


'''6.定义损失函数'''
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

'''7.计算分类准确率'''
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += len(y)
    return acc_sum / n

'''8.定义优化算法(更新权重参数)'''
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

'''9.训练模型'''
# 训练轮数
num_epochs = 5
# 学习率
lr = 0.1

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

# 训练(网络结构，训练集迭代器，测试集迭代器，交叉熵损失函数，训练轮数，batch_size，权重参数，学习率)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

'''9.预测'''
# 训练集
X, y = iter(test_iter).next()
# 真实标签
true_labels = get_fashion_mnist_labels(y.numpy())
# 预测标签
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
# 文本显示
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
# 前十张
show_fashion_mnist(X[0:9], titles[0:9])





