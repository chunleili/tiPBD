# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# %%
# import os
# import sys
 
# path = os.path.abspath(os.path.join(sys.path[0], '..'))+'/'

# total_energy = np.loadtxt("result/log/totalEnergy.txt")
# inertial_energy = np.loadtxt("result/log/inertialEnergy.txt")
# potential_energy = np.loadtxt("result/log/potentialEnergy.txt")

# total_energy = total_energy[0:1000]
# inertial_energy = inertial_energy[0:500]
# potential_energy = potential_energy[0:500]

# %%
total_energy_onlyfine = np.loadtxt("result/log/totalEnergy_onlyfine.txt")
total_energy_onlyfine = total_energy_onlyfine[0:10]

total_energy_mg = np.loadtxt("result/log/totalEnergy_mg.txt")
total_energy_mg = total_energy_mg[0:10]

# %%
seperate = False
plt.figure(figsize=(10, 6))
plt.plot(total_energy_onlyfine, label='total energy only fine', marker='o', markersize=5, color='blue')
if seperate:
    plt.legend()
    plt.figure(figsize=(10, 6))
plt.plot(total_energy_mg, label='total energy mg', marker='x', markersize=5, color='orange')
plt.ylabel('energy')
plt.xlabel('iterations')
plt.legend()
plt.show()