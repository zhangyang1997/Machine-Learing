import numpy as np
import math

N = 1  # 训练集数量
C = 2  # 类别数量

x_train = np.array([0.1, 0.2, 0.3])  # 特征向量
x_train = np.concatenate((x_train, np.ones(1)), axis=0)  # 增广特征向量
y_train = np.array(1)  # 类别标签
w = np.array([[-0.1, 0.2, -0.3, 0.5], [-0.1, 0.2, 0.3, 0.4]],
             dtype=float)  # 增广权重向量


def I(x):#指示函数
    if x > 0:
        return 1
    else:
        return - 1
        
y_predict = I(np.vdot(w[1] - w[0], x_train))

print(y_predict)