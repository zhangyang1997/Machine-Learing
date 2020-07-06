import numpy as np

'''1.数据预处理'''
# 训练样本数量
N = 3
# 学习率
alpha = 0.1
# 特征向量组成的矩阵shape=(d,N)
x_train = np.array([[1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9]], dtype=float)
# 增广特征向量组成的矩阵shape=(d+1,N)
x_train = np.concatenate((x_train, np.ones((1, N))), axis=0)
# 类别标签组成的向量shape=(3,1)
y_train = np.array([[-1],
                    [1],
                    [1]])
# 增广权重向量shape=(d+1,1)
w = np.array([[-1],
              [1],
              [1],
              [1]], dtype=float)

for n in range(N):
    if y_train[n] * np.vdot(w, x_train[:, n]) > 0:
        print("True")
    else:
        print("False")
'''
False
True
True
'''
