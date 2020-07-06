import numpy as np
import math

N = 1  # 训练集数量
C = 3  # 类别数量
x_train = np.array([0.1, 0.2, 0.3])  # 特征向量
x_train = np.concatenate((x_train, np.ones(1)), axis=0)  # 增广特征向量
y_train = np.array(2)  # 类别标签
w = np.array([[-0.1, 0.2, -0.3, 0.5], [-0.1, 0.2, 0.3, 0.4],
              [0.2, 0.3, 0.4, 0.5]], dtype=float)  # 增广权重向量


temp = np.zeros(C)
for c in range(C):
    temp[c] = np.vdot(w[c], x_train)

y_predict = np.argmax(temp)  # 预测类别
print(y_predict)
'''
2
'''
