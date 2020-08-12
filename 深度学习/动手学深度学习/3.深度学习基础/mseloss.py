import torch
import torch.nn as nn
loss = nn.MSELoss()

input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.randn(3, 5)
print(target)
output = loss(input, target)
print(output)

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x ** 2 + 3 * y
print(x.grad, y.grad)
z.backward()
print(x.grad, y.grad)


