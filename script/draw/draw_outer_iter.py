from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

prj_dir = Path(__file__).parent.parent.parent
path = prj_dir / "result/latest/r/n_outer.txt"
print(path)

# draw outer iteration
n_outer = np.loadtxt(path)
# 画出条形图
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.bar(range(len(n_outer)), n_outer, tick_label=range(len(n_outer)))
plt.show()