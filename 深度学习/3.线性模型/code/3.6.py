import numpy as np
import math

w = np.array([1, 2, 3])  # 权重向量
b = 1  # 偏置


def f(x):
    return np.vdot(w, x) + b


gamma = 0  # 特征空间每个样本点到决策平面的有向距离
x = np.array([1, 2, 3])  # 输入
w_norm = np.linalg.norm(w)  # 权重向量的模/2范数

gamma = f(x) / w_norm  # 公式3.6

print(gamma)
'''
4.008918628686366
'''