""" Lowest order edge AMG implementing Reitzinger-Schoberl algorithm"""

import numpy as np
import scipy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import pyamg

from edgeAMG import edgeAMG

Acurl = sparse.csr_matrix(scipy.io.mmread("HCurlStiffness.dat"))
Anode = sparse.csr_matrix(scipy.io.mmread("H1Stiffness.dat"))
D = sparse.csr_matrix(scipy.io.mmread("D.dat"))

ml = edgeAMG(Anode, Acurl, D)
MLOp = ml.aspreconditioner()
x = np.random.rand(Acurl.shape[1], 1)
b = Acurl * x
x0 = np.ones((Acurl.shape[1], 1))

r_edgeAMG = []
r_None = []
r_SA = []

ml_SA = pyamg.smoothed_aggregation_solver(Acurl)
ML_SAOP = ml_SA.aspreconditioner()
x_prec, info = pyamg.krylov.cg(Acurl, b, x0, M=MLOp, tol=1e-10, residuals=r_edgeAMG)
x_prec, info = pyamg.krylov.cg(Acurl, b, x0, M=None, tol=1e-10, residuals=r_None)
x_prec, info = pyamg.krylov.cg(Acurl, b, x0, M=ML_SAOP, tol=1e-10, residuals=r_SA)

fig, ax = plt.subplots()
ax.semilogy(np.arange(0, len(r_edgeAMG)), r_edgeAMG, label='edge AMG')
ax.semilogy(np.arange(0, len(r_None)), r_None, label='CG')
ax.semilogy(np.arange(0, len(r_SA)), r_SA, label='CG + AMG')
ax.grid(True)
plt.legend()

figname = f'./output/edgeAMG_convergence.png'
import sys
if '--savefig' in sys.argv:
    plt.savefig(figname, bbox_inches='tight', dpi=150)
else:
    plt.show()