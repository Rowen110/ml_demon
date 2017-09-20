# -*- coding: utf-8 -*-

# 正态分布概率密度函数

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 视图中显示中文设置
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 均值
mu = 0
# 标准差
sigma = 1
# 均值 - 3倍的标准差， 均值 + 3倍的标准差，  50个元素
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
#
y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.math.sqrt(2 * np.math.pi) * sigma)
print x.shape
print 'x = \n', x
print y.shape
print 'y = \n', y
plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
plt.grid(True)
plt.title(u'Guass分布')
plt.show()
