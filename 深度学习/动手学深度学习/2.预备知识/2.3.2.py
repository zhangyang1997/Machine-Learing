import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = 2 * y
print(z)
print(z.grad_fn)

print(x.is_leaf)
print(y.is_leaf)

z_mean = z.mean()
print(z_mean)

a = torch.ones((2, 2)) + 1
print(a)
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b)
print(b.item())
print(b.grad_fn)
