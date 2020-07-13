import torch
import numpy as np
import matplotlib.pylab as plt
import sys
import IPython.display as display

def use_svg_display():
    '''用矢量图显示'''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    '''设置图的尺寸'''
    # 用矢量图显示
    use_svg_display()
    # 设置尺寸
    plt.rcParams['figure.figsize'] = figsize

def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')

'''激活函数'''
x = torch.arange(-3.0, 3.0, 0.01, requires_grad=True)
# Relu函数
y = x.relu()
xyplot(x, y, 'relu')
plt.show()

y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
plt.show()

# sigmoid函数
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
plt.show()

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')
plt.show()

# tanh函数
y = x.tanh()
xyplot(x, y, 'tanh')
plt.show()

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
plt.show()

