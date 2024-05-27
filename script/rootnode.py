import numpy as np
import pyamg
import sys
import os
import matplotlib.pyplot as plt

from reproduce_pyamg import plot_residuals, to_read_dir, mkdir_if_not_exist, save_fig_instad_of_show, show_plot, load_A_b

A,b = load_A_b('scale64')


fig,ax = plt.subplots(1, figsize=(8, 9))

for solvernum in [1, 2]:
    for accel in [None, 'cg']:
        # Create matrix and candidate vectors.  B has 3 columns, representing
        # rigid body modes of the mesh. B[:,0] and B[:,1] are translations in
        # the X and Y directions while B[:,2] is a rotation.
        # A, B = pyamg.gallery.linear_elasticity((200, 200), format='bsr')


        # Construct solver using AMG based on Smoothed Aggregation (SA)
        if solvernum == 1:
            ml = pyamg.smoothed_aggregation_solver(A, smooth='energy')
        elif solvernum == 2:
            ml = pyamg.rootnode_solver(A, smooth='energy')
        else:
            raise ValueError("Enter a solver of 1 or 2")

        # Display hierarchy information
        print(ml)

        # Create random right hand side
        b = np.random.rand(A.shape[0], 1)

        # Solve Ax=b
        residuals = []
        x = ml.solve(b, tol=1e-10, residuals=residuals, accel=accel)
        print("Number of iterations:  {}d\n".format(len(residuals)))

        # Output convergence
        # for i, r in enumerate(residuals):
        #     print("residual at iteration {0:2}: {1:^6.2e}".format(i, r))

        if solvernum==1 and accel==None:
            plot_residuals(residuals/residuals[0], ax, label=f"{solvernum}:SA {accel}", marker="o", color="blue")
        elif solvernum==1 and accel=='cg':
            plot_residuals(residuals/residuals[0], ax, label=f"{solvernum}:SA {accel}", marker="x", color="blue")
        elif solvernum==2 and accel==None:
            plot_residuals(residuals/residuals[0], ax, label=f"{solvernum}:rootnode {accel}", marker="o", color="red")
        else:
            plot_residuals(residuals/residuals[0], ax, label=f"{solvernum}:rootnode {accel}", marker="x", color="red")
plt.show()