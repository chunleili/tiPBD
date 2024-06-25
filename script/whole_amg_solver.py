import pyamg
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
# from pyamg.krylov import cg
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial
from time import perf_counter
from scipy.linalg import pinv

smoother = 'gauss_seidel'


def setup_chebyshev(lvl, lower_bound=1.0/30.0, upper_bound=1.1, degree=3,
                    iterations=1):
    """Set up Chebyshev."""
    rho = approximate_spectral_radius(lvl.A)
    a = rho * lower_bound
    b = rho * upper_bound
    # drop the constant coefficient
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]

    def chebyshev(A, x, b):
        polynomial(A, x, b, coefficients=coefficients, iterations=iterations)
    return chebyshev



def build_Ps(A, method='UA'):
    """Build a list of prolongation matrices Ps from A """
    if method == 'UA' or method == 'UA_CG':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None)
    elif method == 'SA' or method == 'SA_CG':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400)
    elif method == 'UA_CG_GS':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, coarse_solver='gauss_seidel')
    elif method == 'CAMG' or method == 'CAMG_CG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400)
    else:
        raise ValueError(f"Method {method} not recognized")

    Ps = []
    for i in range(len(ml.levels)-1):
        Ps.append(ml.levels[i].P)

    return Ps


class MultiLevel:
    A = None
    P = None
    R = None


def build_levels(A, Ps=[]):
    '''Give A and a list of prolongation matrices Ps, return a list of levels'''
    lvl = len(Ps) + 1 # number of levels

    levels = [MultiLevel() for i in range(lvl)]

    levels[0].A = A

    for i in range(lvl-1):
        levels[i].P = Ps[i]
        levels[i].R = Ps[i].T
        levels[i+1].A = Ps[i].T @ levels[i].A @ Ps[i]

    return levels


def amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    x = x0.copy()
    A = levels[0].A
    residuals = np.zeros(maxiter+1)
    def psolve(b):
        x = x0.copy()
        V_cycle(levels, 0, x, b)
        # V_cycle_norecur(levels, 0, x, b)
        return x
    bnrm2 = np.linalg.norm(b)
    atol = tol * bnrm2
    r = b - A@(x)
    rho_prev, p = None, None
    normr = np.linalg.norm(r)
    residuals[0] = normr
    for iteration in range(maxiter):
        if normr < atol:  # Are we done?
            break
        z = psolve(r)
        rho_cur = np.dot(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]
        q = A@(p)
        alpha = rho_cur / np.dot(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur
        normr = np.linalg.norm(r)
        residuals[iteration+1] = normr
    residuals = residuals[:iteration+1]
    return (x),  residuals         


# def diag_sweep(A,x,b,iterations=1):
#     Ap = A.indptr
#     Aj = A.indices
#     Ax = A.data
#     for i in range(0, A.shape[0]):
#         start = Ap[i]
#         end = Ap[i + 1]
#         for jj in range(start, end):
#             j = Aj[jj]
#             if i == j:
#                 diag = Ax[jj]
#         if diag != 0.0:
#             x[i] = (b[i]) / diag

def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag


def presmoother(A,x,b):
    if smoother == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif smoother == 'jacobi':
        jacobi(A,x,b,iterations=10)
    elif smoother == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif smoother == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif smoother == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif smoother == 'chebyshev':
        chebyshev(A,x,b)


def postsmoother(A,x,b):
    presmoother(A,x,b)


def V_cycle(levels,lvl,x,b):
    A = levels[lvl].A
    presmoother(A,x,b)
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        V_cycle(levels, lvl+1, coarse_x, coarse_b)
    x += levels[lvl].P @ coarse_x
    postsmoother(A, x, b)


def V_cycle_norecur(levels,lvl,x,b):
    lvl = 0
    x0=x
    b0=b
    A0 = levels[lvl].A
    gauss_seidel(A0,x0,b0,iterations=1, sweep='symmetric') # presmoother
    residual0 = b0 - A0 @ x0
    b1 = levels[lvl].R @ residual0

    lvl = 1
    x1 = np.zeros_like(b1)
    A1 = levels[lvl].A
    gauss_seidel(A1,x1,b1,iterations=1, sweep='symmetric') # presmoother
    residual1 = b1 - A1 @ x1
    b2 = levels[lvl].R @ residual1

    lvl = 2
    A2 = levels[lvl].A
    x2 = coarse_solver(A2, b2)

    lvl = 1
    x1 += levels[lvl].P @ x2
    gauss_seidel(A1,x1,b1,iterations=1, sweep='symmetric') # postsmoother

    lvl = 0
    x0 += levels[lvl].P @ x1
    gauss_seidel(A0,x0,b0,iterations=1, sweep='symmetric') # postsmoother

    x = x0


# 实现仅第一次进入coarse_solver时计算一次P
# https://stackoverflow.com/a/279597/19253199
def coarse_solver(A, b):
    if not hasattr(coarse_solver, "P"):
        coarse_solver.P = pinv(A.toarray())
    res = np.dot(coarse_solver.P, b)
    return res


def demo_my_own_mg():
    import os, sys
    sys.path.append(os.getcwd())
    from utils.load_A_b import load_A_b
    from utils.solvers import UA_CG, UA_CG_chebyshev, UA_CG_jacobi, CG
    from collections import namedtuple
    from utils.plot_residuals import plot_residuals_all
    from utils.postprocess_residual import print_allres_time
    from utils.parms import maxiter
    Residual = namedtuple('Residual', ['label','r', 't'])
    global smoother, chebyshev

    A, b = load_A_b('F30-0')

    # for _ in range(5):
    t0= perf_counter()
    Ps = build_Ps(A)
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    print('Setup Time:', t1-t0)


    chebyshev = setup_chebyshev(levels[0], lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)


    smoother = 'diag_sweep'
    x0 = np.zeros_like(b)
    t2= perf_counter()
    residuals = []
    x,residuals = amg_cg_solve(levels, b, x0=x0, maxiter=maxiter, tol=1e-6)
    t3 = perf_counter()
    print('My diag_sweep Time:', t3-t2)
    # print('Total Time:', t3-t2+t1-t0)
    allres = [Residual('diag_sweep', residuals, t3-t0)]

    tic = perf_counter()
    UA_CG(A, b, x0, allres)
    toc = perf_counter()
    print("UA_CG Time:", toc-tic)

    tic = perf_counter()
    UA_CG_chebyshev(A, b, x0, allres)
    toc = perf_counter()
    print("UA_CG_chebyshev Time:", toc-tic)

    tic = perf_counter()
    UA_CG_jacobi(A, b, x0, allres)
    toc = perf_counter()
    print("UA_CG_jacobi Time:", toc-tic)

    tic = perf_counter()
    CG(A, b, x0, allres)
    toc = perf_counter()
    print("CG Time:", toc-tic)

    smoother = 'sor'
    tic = perf_counter()
    x,residuals = amg_cg_solve(levels, b, x0=x0, maxiter=maxiter, tol=1e-6)
    toc = perf_counter()
    print("My sor Time:", toc-tic)
    allres.append(Residual('sor', residuals, toc-tic))


    smoother = 'chebyshev'
    tic = perf_counter()
    x,residuals = amg_cg_solve(levels, b, x0=x0, maxiter=maxiter, tol=1e-6)
    toc = perf_counter()
    print("My chebyshev Time:", toc-tic)
    allres.append(Residual('chebyshev', residuals, toc-tic))

    print_allres_time(allres, draw=True)

    plot_residuals_all(allres,use_markers=True)


if __name__ == "__main__":
    demo_my_own_mg()