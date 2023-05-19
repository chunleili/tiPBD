# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_frame", "-f", type=int, default=10)
max_frame = parser.parse_args().max_frame

# %%
total_energy_onlyfine = np.loadtxt("result/log/totalEnergy_onlyfine.txt")
total_energy_onlyfine = total_energy_onlyfine[0:max_frame]

total_energy_mg = np.loadtxt("result/log/totalEnergy_mg.txt")
total_energy_mg = total_energy_mg[0:max_frame]

# %%
seperate = False
plt.figure(figsize=(10, 6))
plt.plot(total_energy_onlyfine, label="total energy only fine", marker="o", markersize=5, color="blue")
if seperate:
    plt.legend()
    plt.figure(figsize=(10, 6))
plt.plot(total_energy_mg, label="total energy mg", marker="x", markersize=5, color="orange")
plt.ylabel("energy")
plt.xlabel("iterations")
plt.legend()
plt.show()
