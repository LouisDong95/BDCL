import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.arange(0.1, 1.0, 0.2)
results = pd.read_csv('../results/Fashion_tau.csv')

acc = results['ACC']
nmi = results['NMI']

# 绘图
plt.plot(x, acc, color='blue', label='ACC')  # 蓝色线
plt.plot(x, nmi, color='red', label='NMI')   # 红色线

# 添加图例
plt.legend()

# 添加标题和标签
plt.ylim(0.8, 1.0)
plt.xlabel(r'$\tau$')
plt.ylabel('')
x_major_locator = plt.MultipleLocator(0.2)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xticks(x)

plt.savefig("./results/Fashion_tau.pdf", bbox_inches='tight', pad_inches=0)
