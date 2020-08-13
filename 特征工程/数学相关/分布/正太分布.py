from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
# 正态分布（normal distribution）
fig, ax = plt.subplots(1, 1)

loc = 1
scale = 2.0

# ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.linspace(norm.ppf(0.01, loc, scale), norm.ppf(0.99, loc, scale), 100)
ax.plot(x, norm.pdf(x, loc, scale), '-', label='norm')

plt.title(u'normal distribution')
plt.show()