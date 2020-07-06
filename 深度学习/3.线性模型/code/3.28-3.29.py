import numpy as np
import math

N = 1  # 训练集数量
C = 3  # 类别数量
x_train = np.array([0.1, 0.2, 0.3])  # 特征向量
x_train = np.concatenate((x_train, np.ones(1)), axis=0)  # 增广特征向量
y_train = np.array(2)  # 类别标签
w = np.array([[-0.1, 0.2, -0.3, 0.5], [-0.1, 0.2, 0.3, 0.4],
              [0.2, 0.3, 0.4, 0.5]], dtype=float)  # 增广权重向量


def softmax(w_c_x):  # Softmax函数
    sum = 0
    for c in range(C):
        sum += math.exp(np.vdot(w[c], x_train))
    return math.exp(w_c_x) / sum


p_c_x = np.zeros(C)  # 预测属于类别c的条件概率
for c in range(C):
    p_c_x[c] = softmax(np.vdot(w[c], x_train))

print(p_c_x)
'''
[0.29583898 0.32047854 0.38368248]
'''
