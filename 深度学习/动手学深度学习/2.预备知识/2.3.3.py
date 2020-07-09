import torch

x = torch.ones((2, 2), requires_grad=True)
print(x)
y = x + 2
print(y)
z = y * y * 3
print(z)
out = z.mean()
print(out)
out.backward()
print(x.grad)

print()

out2 = x.sum()
print(out2)
out2.backward()
print(x.grad)

print()

out3 = x.sum()
x.grad.data.zero_()
print(x.grad)
print(out3)
out3.backward()
print(x.grad)

print()

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
print(x)
y = 2 * x
print(y)
w = torch.tensor([1.0, 0.1, 0.01, 0.001])
print(w)
y.backward(w)
print(x.grad)

x.grad.data.zero_()
y = 2 * x
print(y)
l = torch.sum(y * w)
l.backward()
print(x.grad)

print()

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)
y3.backward()
print(x.grad)

print()

x = torch.ones(1, requires_grad=True)
print(x)
print(x.data)
print(x.item())
print(x.data.requires_grad)
y = 2 * x
y.backward()
print(x.grad)

print()

x.grad.data.zero_()
x.data *= 100
print(x.data)
y = 2 * x
y.backward()
print(x)
print(x.grad)
