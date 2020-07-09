import torch
import numpy as np

'''tensor转array(共享内存)'''
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

print()

'''array转tensor(共享内存)'''
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

print()

'''array转numpy(拷贝)'''
a = np.ones(5)
b = torch.tensor(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)
