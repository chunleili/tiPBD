# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# %%
onlyfine = np.loadtxt("result/log/residual_onlyfine.txt")
onlyfine = onlyfine[0:10]

mg = np.loadtxt("result/log/residual_mg.txt")
mg = mg[0:10]
# 由于mg的fine mesh只在后5个iter里计算，所以前面空出来


# %%
seperate = False
plt.figure(figsize=(10, 6))
plt.plot(onlyfine, label='residual only fine', marker='o', markersize=5, color='blue')
if seperate:
    plt.legend()
    plt.figure(figsize=(10, 6))
plt.plot(mg, label='residual mg', marker='x', markersize=5, color='orange')
plt.ylabel('residual(2-norm)')
plt.xlabel('iterations')
plt.legend()
plt.show()