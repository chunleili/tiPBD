# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onlyfine_start_iter", "-ofs", type=int, default=0)
parser.add_argument("--onlyfine_end_iter", "-ofe", type=int, default=10)
parser.add_argument("--multigrid_start_iter", "-mgs", type=int, default=0)
parser.add_argument("--multigrid_end_iter", "-mge", type=int, default=10)
onlyfine_start_iter = parser.parse_args().onlyfine_start_iter
onlyfine_end_iter = parser.parse_args().onlyfine_end_iter
multigrid_start_iter = parser.parse_args().multigrid_start_iter
multigrid_end_iter = parser.parse_args().multigrid_end_iter
# %%
total_energy_onlyfine = np.loadtxt("result/log/totalEnergy_onlyfine.txt")
total_energy_onlyfine = total_energy_onlyfine[onlyfine_start_iter:onlyfine_end_iter]

total_energy_mg = np.loadtxt("result/log/totalEnergy_mg.txt")
total_energy_mg = total_energy_mg[multigrid_start_iter:multigrid_end_iter]

# %%
seperate = False
plt.figure(figsize=(10, 6))
plt.plot(
    total_energy_onlyfine,
    label="total energy only fine",
    marker="o",
    markersize=5,
    color="blue",
)
if seperate:
    plt.legend()
    plt.figure(figsize=(10, 6))
plt.plot(total_energy_mg, label="total energy mg", marker="x", markersize=5, color="orange")
plt.ylabel("energy")
plt.xlabel("iterations")
plt.legend()
plt.show()
