import torch

'''1.运算开辟新内存'''
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
y = y + x
print(y)
id_after = id(y)
print(id_after)

print()

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
y = torch.add(x, y)
print(y)
id_after = id(y)
print(id_after)

print()

'''2.运算不开辟新内存'''
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
y += x
print(y)
id_after = id(y)
print(id_after)

print()

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
y[:] = y+x
print(y)
id_after = id(y)
print(id_after)

print()

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
torch.add(x, y, out=y)
print(y)
id_after = id(y)
print(id_after)

print()

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
y.add_(x)
print(y)
id_after = id(y)
print(id_after)