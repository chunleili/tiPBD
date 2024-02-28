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
parser.add_argument("--yscale", "-y", type=str, default="log")
parser.add_argument("--xticks", "-xt", nargs=3, type=int, default=None)

args = parser.parse_args()

onlyfine_start_iter, onlyfine_end_iter = args.onlyfine_iter_range
multigrid_start_iter, multigrid_end_iter = args.multigrid_iter_range
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
plt.yscale(args.yscale)
plt.ylabel("residual(2-norm)")
plt.xlabel("iterations")
if args.xticks is not None:
    plt.xticks(np.arange(*args.xticks))
plt.legend()
plt.show()
