# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onlyfine_iter_range", "-of", nargs=2, type=int, default=(0, 10))
parser.add_argument("--multigrid_iter_range", "-mg", nargs=2, type=int, default=(0, 10))

onlyfine_start_iter, onlyfine_end_iter = parser.parse_args().onlyfine_iter_range
multigrid_start_iter, multigrid_end_iter = parser.parse_args().multigrid_iter_range
# %%
onlyfine = np.loadtxt("result/log/residual_onlyfine.txt")
onlyfine = onlyfine[onlyfine_start_iter:onlyfine_end_iter]

mg = np.loadtxt("result/log/residual_mg.txt")
mg = mg[multigrid_start_iter:multigrid_end_iter]
# mg[0:5] = np.nan


# %%
seperate = False
plt.figure(figsize=(10, 6))
plt.plot(onlyfine, label="residual only fine", marker="o", markersize=5, color="blue")
if seperate:
    plt.legend()
    plt.figure(figsize=(10, 6))
plt.plot(mg, label="residual mg", marker="x", markersize=5, color="orange")
plt.ylabel("residual(2-norm)")
plt.xlabel("iterations")
plt.legend()
plt.show()
