import torch
from torch import nn


class MySequential(nn.Module):
    from collections import OrderedDict

    def __init__(self, *args):
        super(MySequential, self).__init__()
        # 如果传入的是一个OrderedDict
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                # add_module方法会将module添加进self._modules(一个OrderedDict)
                self.add_module(key, module)
         # 如果传入的是一些Module
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


X = torch.rand(2, 784)
net = MySequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10),)
print(net)
output = net(X)
print(output.shape)
print()