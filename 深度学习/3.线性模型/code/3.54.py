import numpy as np


def sgn(x):
    '''
    符号函数
    '''
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


# 输入特征向量
x = np.array([[1],
              [2],
              [3]], dtype=float)
# 增广特征向量
x = np.concatenate((x, np.ones((1, 1))), axis=0)
# 增广权重向量
w = np.array([[1],
              [2],
              [3],
              [1]], dtype=float)
# 预测类别标签
y_predict = sgn(np.vdot(w, x))

print(y_predict)
'''
1
'''
