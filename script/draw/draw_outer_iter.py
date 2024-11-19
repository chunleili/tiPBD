from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

prj_dir = Path(__file__).parent.parent.parent
path = prj_dir / "result/case148-1106-XPBD-strain/r/n_outer.txt"
print(path)

# draw outer iteration
n_outer = np.loadtxt(path)

# 画出条形图

# 设置 x 轴刻度间隔为 10
fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(10))
# ax..yaxis.set_major_locator(MaxNLocator(integer=True))
ax.bar(range(len(n_outer)), n_outer)
plt.show()