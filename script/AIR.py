# AIR Example
import numpy as np
import pyamg
import sys
import matplotlib.pyplot as plt

from reproduce_pyamg import plot_residuals, load_A_b

A,b = load_A_b('scale64')

# ax = list(ax.flatten())
i=0
# Construct AIR solver
for dist in [1,2]:
    for second_pass in  [False, True]:
        if second_pass:
            if dist == 1:
                print('Distance-1 AIR using RS coarsening *with* second pass.')
            else:
                print('Distance-2 AIR using RS coarsening *with* second pass.')
        else:
            if dist == 1:
                print('Distance-1 AIR using RS coarsening *without* second pass.')
            else:
                print('Distance-2 AIR using RS coarsening *without* second pass.')

        # Specify restriction and coarsening
        restrict=('air', {'theta': 0.1, 'degree': dist})
        CF =('RS', {'second_pass': second_pass})
        ml = pyamg.air_solver(A, CF=CF, restrict=restrict)

        # Solve Ax=b
        residuals = []
        x = ml.solve(b, tol=1e-10, accel=None, residuals=residuals)
        conv = (residuals[-1]/residuals[0])**(1.0/(len(residuals)-1))
        print(f'\tLevels in hierarchy:        {len(ml.levels)}')
        print(f'\tOperator complexity:        {ml.operator_complexity()}')
        print(f'\tNumber of iterations:       {len(residuals)-1}')
        print(f'\tAverage convergence factor: {conv}\n')

        fig, ax = plt.subplots(1)
        plot_residuals(residuals, ax, label=f'AIR: dist={dist}, second_pass={second_pass}', marker='o')
        i+=1
        plt.show()

# nx = 50
# ny = 50
# theta = np.pi/6.0
# A, b = pyamg.gallery.advection_2d((ny,nx), theta)
# restrict=('air', {'theta': 0.1, 'degree': 1})
# CF =('RS', {'second_pass': True})

# ml = pyamg.air_solver(A, CF=CF, restrict=restrict)
# xx = np.linspace(0,1,nx-1)
# x,y = np.meshgrid(xx,xx)
# V = np.concatenate([[x.ravel()],[y.ravel()]],axis=0).T
# splitting = ml.levels[0].splitting
# F = np.where(splitting == 1)[0]

