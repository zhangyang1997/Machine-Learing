import numpy as np

C = 3  # 类别数
y_train = 1  # 一个训练样本标签
y_train_v = np.zeros(C)  # 标签对应的one-hot向量
y_train_v[y_train] = 1

print(y_train_v)
'''
[0. 1. 0.]
'''