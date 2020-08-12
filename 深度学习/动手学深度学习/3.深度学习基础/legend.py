"""
===============================
使用预定义标签的图例
===============================

用图形定义图例标签。
"""


import numpy as np
import matplotlib.pyplot as plt

# 制造一些假数据。
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]

# 创建带有预定义标签的图形。
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c + d, 'k', label='Total message length')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# 在图例上放一个更好的背景颜色。
legend.get_frame().set_facecolor('C0')

plt.show()

#############################################################################
#
# ------------
#
# 参考
# """"""""""
#
# 本例中显示了以下函数、方法、类和模块的使用

import matplotlib
matplotlib.axes.Axes.plot
matplotlib.pyplot.plot
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
