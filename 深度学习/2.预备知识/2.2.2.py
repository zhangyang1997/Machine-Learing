import torch

'''算术'''
# 加法形式1
x = torch.ones(5, 3)
y = torch.ones((5, 3))
z = x + y
print(z)
# 加法形式2
z = torch.add(x, y)
print(z)
# 加法形式3
z = x.add_(y)
print(z)

'''索引'''
y = x[0, :]
print(y)
y += 1
print(y)
print(x[0, :])

'''改变形状(共享内存)'''
y = x.reshape(15)
print(y)
y += 1
print(y)
print(x)
z = x.reshape((-1, 5))
print(z)
print()
# 拷贝
y = x.clone()
print(y)
y += 1
print(y)
print(x)
# 取值
x = torch.randn(1)
print(x)
print(x.item())

'''线性代数'''
X = torch.tensor([[1, 2, 0],
                  [0, 3, 4],
                  [0, 0, 5]], dtype=float)
print(x)
# 矩阵的迹
X_trace = torch.trace(X)
print(X_trace)
# 对角线元素组成的向量
X_diag = torch.diag(X)
print(X_diag)
# 矩阵乘法
Y = torch.mm(X, X)
print(Y)
# 矩阵转置
Y = torch.t(X)
print(Y)
# 向量内积
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.dot(x, y)
print(z)
# 向量外积
z = torch.cross(x, y)
print(z)
# 求逆矩阵
X_inverse = torch.inverse(X)
print(X_inverse)
# 奇异值分解
X = torch.tensor([[1, 2, 0],
                  [0, 3, 4],
                  [0, 0, 5],
                  [1, 0, 0]], dtype=float)
X_svd = torch.svd(X)
print(X_svd)
