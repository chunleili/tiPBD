# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# %%
import os
import sys
 
path = os.path.abspath(os.path.join(sys.path[0], '..'))+'/'

total_energy = np.loadtxt("result/log/totalEnergy.txt")
inertial_energy = np.loadtxt("result/log/inertialEnergy.txt")
potential_energy = np.loadtxt("result/log/potentialEnergy.txt")
total_energy.shape
inertial_energy.shape
potential_energy.shape

total_energy = total_energy[0:500]
inertial_energy = inertial_energy[0:500]
potential_energy = potential_energy[0:500]

# %%
plt.figure(figsize=(10, 6))
plt.plot(total_energy, label='total energy', marker='o', markersize=2)
plt.plot(inertial_energy, label='inertial energy')
plt.plot(potential_energy, label='potential energy')
plt.ylabel('energy')
plt.xlabel('iterations')
plt.legend()
plt.show()