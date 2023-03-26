# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# %%
import os
import sys
 
path = os.path.abspath(os.path.join(sys.path[0], '..'))+'/'

total_energy = np.loadtxt(path+'totalEnergy.txt')
kinetic_energy = np.loadtxt(path+'kineticEnergy.txt')
potential_energy = np.loadtxt(path+'potentialEnergy.txt')

total_energy.shape
kinetic_energy.shape
potential_energy.shape

# %%
plt.figure(figsize=(10, 6))
plt.plot(total_energy, label='total energy', marker='o', markersize=2)
plt.plot(kinetic_energy, label='kinetic energy')
plt.plot(potential_energy, label='potential energy')
plt.legend()
plt.show()