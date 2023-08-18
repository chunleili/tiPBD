# import pyamg
# import numpy as np
from scipy.io import mmread, mmwrite
# # A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
# A = mmread("A.mtx")
# ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
# print(ml)                                           # print hierarchy information
# b = np.random.rand(A.shape[0])                      # pick a random right hand side
# x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-10
# print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector




# manual construction of a two-level AMG hierarchy
from pyamg.gallery import poisson
from pyamg.multilevel import MultilevelSolver
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.interpolate import direct_interpolation
from pyamg.classical.split import RS
# compute necessary operators
A = poisson((100, 100), format='csr')
C = classical_strength_of_connection(A)
splitting = RS(A)
P = direct_interpolation(A, C, splitting)
R = P.T
# store first level data
levels = []
levels.append(MultilevelSolver.Level())
levels.append(MultilevelSolver.Level())
levels[0].A = A
levels[0].C = C
levels[0].splitting = splitting
levels[0].P = P
levels[0].R = R
# store second level data
levels[1].A = R @ A @ P                      # coarse-level matrix
# create MultilevelSolver
ml = MultilevelSolver(levels, coarse_solver='splu')
print(ml)