import torch

# 创建未初始化Tensor
x = torch.empty((5, 3))
print(x)
# 创建随机初始化Tensor
x = torch.rand((5, 3))
print(x)
# 创建全零Tensor
x = torch.zeros((5, 3), dtype=torch.long)
print(x)
# 创建有初始化Tensor
x = torch.tensor([5.5, 3])
print(x)
# 返回的tensor默认具有相同的torch.dtype和torch.device
x = x.new_ones((5, 3), dtype=torch.float64)
print(x)
# 指定新的数据类型
x = torch.rand_like(x, dtype=torch.float)
print(x)
# 返回Tensor的形状
print(len(x), len(x[0]))
print(x.size())
print(x.shape)

# 创建Tensor
x = torch.tensor([5.5, 3])
print(x)
# 全1Tensor
x = torch.ones((3, 4))
print(x)
# 全0Tensor
x = torch.zeros((3, 4))
print(x)
# 对角线为1，其他为0
x = torch.eye(3)
print(x)
# 从s到e(不包括e)，步长为step
x = torch.arange(0, 11, 2)
print(x)
# 从s到e(包括e)，均匀切分成steps份
x = torch.linspace(0, 10, 6)
print(x)
# 随机初始化，正态分布
x = torch.rand((3, 4))
print(x)
# 随机初始化，标准正态分布
x = torch.randn((3, 4))
print(x)
# 随机初始化，正态分布
mean = torch.zeros((3, 4))
print(mean)
x = torch.normal(mean, std=1)
print(x)
# 随机排列
per = torch.randperm(len(x))
x = x[per, :]
print(x)
