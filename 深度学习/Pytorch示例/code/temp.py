import torch

x = torch.tensor([[-1, 2, 3]])
y = x.clamp(min=0)
print(y)