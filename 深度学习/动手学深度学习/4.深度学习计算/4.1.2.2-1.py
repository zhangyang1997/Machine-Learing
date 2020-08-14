import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

X = torch.rand(2, 10)
net = MyModule()
print(net)
output = net(X)
print(output.shape)
print()