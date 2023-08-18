# ------------------------------------------------------------------
# Step 1: import scipy and pyamg packages
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import pyamg
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.gallery import poisson

# ------------------------------------------------------------------
# Step 2: setup up the system using pyamg.gallery
# ------------------------------------------------------------------
# n = 50
# X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
# stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=np.pi / 3)
# A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
# b = np.random.rand(A.shape[0])                     # pick a random right hand side
# print("A.shape: ", A.shape)


A = poisson((20, 20), format="csr")
b = np.random.rand(A.shape[0])  # pick a random right hand side

# ------------------------------------------------------------------
# Step 3: setup of the multigrid hierarchy
# ------------------------------------------------------------------
# ml = pyamg.smoothed_aggregation_solver(A)   # construct the multigrid hierarchy
ml = pyamg.classical.ruge_stuben_solver(A, max_levels=2)  # construct the multigrid hierarchy
print("ml bulit")

# ------------------------------------------------------------------
# Step 4: solve the system
# ------------------------------------------------------------------
res1 = []
x = ml.solve(b, tol=1e-6, residuals=res1)  # solve Ax=b to a tolerance of 1e-12

# ------------------------------------------------------------------
# Step 5: print details
# ------------------------------------------------------------------
print("\n")
print("Details: Default AMG")
print("--------------------")
print(ml)  # print hierarchy information

print("The residual norm is {}".format(np.linalg.norm(b - A * x)))  # compute norm of residual vector


def print_details(ml):
    # notice that there are 5 (or maybe 6) levels in the hierarchy
    #
    # we can look at the data in each of the levels
    # e.g. the multigrid components on the finest (0) level
    #      A: operator on level 0
    #      P: prolongation operator mapping from level 1 to level 0
    #      R: restriction operator mapping from level 0 to level 1
    #      B: near null-space modes for level 0
    #      presmoother: presmoothing function taking arguments (A,x,b)
    #      postsmoother: postsmoothing function taking arguments (A,x,b)
    print("\n")
    print("Details of Multigrid Hierarchy")
    print("-----------------------")
    for l in range(len(ml.levels)):
        An = ml.levels[l].A.shape[0]
        Am = ml.levels[l].A.shape[1]
        if l == (len(ml.levels) - 1):
            print(f"A_{l}: {An:>10}x{Am:<10}")
        else:
            Pn = ml.levels[l].P.shape[0]
            Pm = ml.levels[l].P.shape[1]
            print(f"A_{l}: {An:>10}x{Am:<10}   P_{l}: {Pn:>10}x{Pm:<10}")


print_details(ml)

# ------------------------------------------------------------------
# Step 6: change the hierarchy
# ------------------------------------------------------------------
# we can also change the details of the hierarchy
ml = pyamg.smoothed_aggregation_solver(
    A,  # the matrix
    BH=None,  # the representation of the left near null space
    symmetry="hermitian",  # indicate that the matrix is Hermitian
    strength="evolution",  # change the strength of connection
    aggregate="standard",  # use a standard aggregation method
    smooth=("jacobi", {"omega": 4.0 / 3.0, "degree": 2}),  # prolongation smoothing
    presmoother=("block_gauss_seidel", {"sweep": "symmetric"}),
    postsmoother=("block_gauss_seidel", {"sweep": "symmetric"}),
    improve_candidates=[("block_gauss_seidel", {"sweep": "symmetric", "iterations": 4}), None],
    max_levels=10,  # maximum number of levels
    max_coarse=5,  # maximum number on a coarse level
    keep=False,
)  # keep extra operators around in the hierarchy (memory)

# ------------------------------------------------------------------
# Step 7: print details
# ------------------------------------------------------------------
res2 = []  # keep the residual history in the solve
x = ml.solve(b, tol=1e-6, residuals=res2)  # solve Ax=b to a tolerance of 1e-12
print("\n")
print("Details: Specialized AMG")
print("------------------------")
print(ml)  # print hierarchy information
print("The residual norm is {}".format(np.linalg.norm(b - A * x)))  # compute norm of residual vector
print("\n")


res3 = []
# # ---------------------------------------------------------------------------- #
# #                                My specialized                                #
# # ---------------------------------------------------------------------------- #

ml = pyamg.classical.ruge_stuben_solver(
    A,
    max_levels=2,
    presmoother=("gauss_seidel", {"sweep": "symmetric"}),
    postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
)  # construct the multigrid hierarchy
print("ml bulit")
N = A.shape[0]
# ml.levels[0].R = np.random.rand(int(N/2), N, dtype=bool)
ml.levels[0].R = sparse.random(int(N / 2), N, density=0.001, dtype=bool)
ml.levels[0].P = ml.levels[0].R.transpose()
smoothers = ("gauss_seidel", {"iterations": 0})
change_smoothers(ml, presmoother=None, postsmoother=smoothers)
x = ml.solve(b, tol=1e-6, residuals=res3)
print("\n")
print("Details: My AMG")
print("--------------------")
print(ml)  # print hierarchy information
print_details(ml)
print("Final residual norm is {}".format(np.linalg.norm(b - A * x)))  # compute norm of residual vector

# # ---------------------------------------------------------------------------- #
# #                              End My Specialized                              #
# # ---------------------------------------------------------------------------- #


# ------------------------------------------------------------------
# Step 8: plot convergence history
# ------------------------------------------------------------------
fig, ax = plt.subplots()
ax.semilogy(res1, label="Default AMG solver")
ax.semilogy(res2, label="Specialized AMG solver")
# ax.semilogy(res3, "*",label='my AMG solver')
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Residual")
ax.grid(True)
plt.legend()

figname = f"./output/amg_convergence.png"
import sys

if "--savefig" in sys.argv:
    plt.savefig(figname, bbox_inches="tight", dpi=150)
else:
    plt.show()
