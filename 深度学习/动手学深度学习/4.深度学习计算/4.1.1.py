import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        print("hidden",self.hidden)
        self.act = nn.ReLU()
        print("act",self.act)
        self.output = nn.Linear(256, 10)
        print("output",self.output)
        print()

    def forward(self, x):
        a = self.act(self.hidden(x))
        print("a.shape",a.shape)
        print()
        return self.output(a)
    
X = torch.rand(2, 784)
print("X.shape",X.shape)
net = MLP()
print("net",net)
output = net(X)
print("output.shape", output.shape)
print("output",output)

'''
net(X)会调用MLP继承自Module类的__call__函数，
这个函数将调用MLP类定义的forward函数来完成前向计算。
'''


