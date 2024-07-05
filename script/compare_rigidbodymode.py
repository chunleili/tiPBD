import pyamg
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from pyamg.krylov import cg
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial
from time import perf_counter
from scipy.linalg import pinv

smoother = 'gauss_seidel'
update_coarse_solver = False

from utils.build_Ps import build_Ps
from utils.build_levels import build_levels
from utils.smoothers import presmoother, postsmoother, chebyshev
from utils.coarse_solver import coarse_solver
from utils.amg_cg_solve import amg_cg_solve



def main(postfix='F10-0'):
    import os, sys
    sys.path.append(os.getcwd())
    from utils.load_A_b import load_A_b
    from utils.solvers import UA_CG, UA_CG_chebyshev, UA_CG_jacobi, CG, diagCG
    from collections import namedtuple
    from utils.plot_residuals import plot_residuals_all
    from utils.postprocess_residual import print_allres_time, calc_conv, print_df_newnew
    from utils.parms import maxiter
    Residual = namedtuple('Residual', ['label','r', 't'])
    global smoother, chebyshev, levels


    A, b = load_A_b(postfix)
    x0 = np.zeros_like(b)

    allres = []

    UA_CG(A,b,x0,allres)


    t0= perf_counter()
    Ps = build_Ps(A, method='adaptive_SA')
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    print('Setup Time:', t1-t0)
    print(f"levels:{len(levels)}")
    for i in range(len(levels)):
        print(f"level {i} shape: {levels[i].A.shape}")
    for i in range(len(levels)-1):
        ratio = levels[i].A.shape[0] / levels[i+1].A.shape[0]
        print(f"level {i} ratio: {ratio}")

    smoother = 'gauss_seidel'
    tic = perf_counter()
    x0 = np.zeros_like(b)
    x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=maxiter, tol=1e-6)
    toc = perf_counter()
    print("solve phase:", toc-tic)
    allres.append(Residual('adpativeSA', residuals, toc-tic))
    conv = calc_conv(residuals)
    print("conv:", conv)
    print("iterations:", len(residuals)-1)

    print_df_newnew(allres)
    plot_residuals_all(allres,use_markers=True)


if __name__ == "__main__":
    # for f in [10,20,30,40,50,60]:
    for f in [10]:
        postfix = f"F{f}-0"
        print(f"Postfix: {postfix}")
        main(postfix)
