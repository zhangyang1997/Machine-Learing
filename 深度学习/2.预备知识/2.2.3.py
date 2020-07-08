import torch

x = torch.arange(1, 3).view((1,2))
print(x)

y = torch.arange(1, 4).view((3,1))
print(y)

z = x + y
print(z)
'''
tensor([[1, 2]])
tensor([[1],
        [2],
        [3]])
tensor([[2, 3],
        [3, 4],
        [4, 5]])
'''