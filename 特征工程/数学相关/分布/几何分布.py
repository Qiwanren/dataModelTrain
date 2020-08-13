from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt
# 几何分布（geometric distribution）
n = 10
p = 0.5
k = np.arange(1,10)
geom_dis = geom.pmf(k,p)
plt.plot(k, geom_dis, 'o-')
plt.title('geometric distribution')
plt.xlabel('i-st item success')
plt.ylabel('probalility of i-st item success')
plt.grid(True)
plt.show()