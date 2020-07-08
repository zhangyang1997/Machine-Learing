import numpy as np

# 滤波器
w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
K = len(w)
# 信号序列
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
M = len(x)
# 累积信号序列
y = np.zeros_like(x)
# 卷积即加权求和
for t in range(K-1, M):
    for k in range(K):
        y[t] += w[k] * x[t - k]

print(y)
'''
[0. 0. 2. 3. 4. 5. 6. 7. 8.]
'''
