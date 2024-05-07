import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = 'Fashion'
metrics = 'ACC'

COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]
lambda1 = lambda2 = [0.01, 0.1, 1, 10, 100]
x = list(range(len(lambda1)))
y = list(range(len(lambda2)))
x_tickets = [str(_x) for _x in lambda1]
y_tickets = [str(_y) for _y in lambda2]

data = pd.read_csv('./results/%s_result.csv' %dataset)
acc = data[metrics].to_numpy().reshape(len(x), len(y))

xx, yy = np.meshgrid(x, y)

color_list = []
for i in range(len(y)):
    c = COLOR[i]
    color_list.append([c] * len(x))
color_list = np.asarray(color_list)

xx_flat, yy_flat, acc_flat, color_flat = xx.ravel(), yy.ravel(), acc.ravel(), color_list.ravel()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.bar3d(xx_flat - 0.35, yy_flat - 0.35, 0, 0.7, 0.7, acc_flat,
         color=color_flat,  # 颜色
         edgecolor="black",  # 黑色描边
         shade=True)  # 加阴影

# 座标轴名
ax.set_xlabel(r"$\lambda_2$")
ax.set_ylabel(r"$\lambda_1$")
ax.set_zlabel(metrics)

# 座标轴范围
ax.set_zlim((0, 1.01))


ax.set_xticks(y)
ax.set_xticklabels(y_tickets)
ax.set_yticks(x)
ax.set_yticklabels(x_tickets)

# 保存
# plt.tight_layout()
fig.savefig("./results/%s_%s.svg" %(dataset, metrics), bbox_inches='tight', pad_inches=0)
plt.close(fig)