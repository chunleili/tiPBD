import pyamg
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
# from pyamg.krylov import cg
from pyamg.relaxation.relaxation import gauss_seidel
from time import perf_counter
from scipy.linalg import pinv


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


def V_cycle(levels,lvl,x,b):
    A = levels[lvl].A
    gauss_seidel(A,x,b,iterations=1, sweep='symmetric') # presmoother
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        V_cycle(levels, lvl+1, coarse_x, coarse_b)
    x += levels[lvl].P @ coarse_x
    gauss_seidel(A,x,b,iterations=1, sweep='symmetric')


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
    from utils.solvers import UA_CG
    from collections import namedtuple
    from utils.plot_residuals import plot_residuals_all
    Residual = namedtuple('Residual', ['label','r', 't'])

    A, b = load_A_b('F11-0')
    t0= perf_counter()
    Ps = build_Ps(A)
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    print('Setup Time:', t1-t0)

    x0 = np.zeros_like(b)
    t2= perf_counter()
    residuals = []
    x,residuals = amg_cg_solve(levels, b, x0=x0, maxiter=100, tol=1e-6)
    # x = _amg_cg_solve(levels, b, x0=x0, maxiter=100, tol=1e-6, residuals=residuals)
    t3 = perf_counter()
    print('My Solve Time:', t3-t2)
    # print('Total Time:', t3-t2+t1-t0)
    allres = [Residual('MyOwnMG', residuals, t3-t0)]

    tic = perf_counter()
    UA_CG(A, b, x0, allres)
    toc = perf_counter()
    print("UA_CG Time:", toc-tic)

    # sys.path.append("C:\dev\tiPBD\pybind\build\Debug")
    import amg_cg_solve_bind
    # amg_cg_solve_bind.amg_cg_solve(levels, b, x0=x0, maxiter=100, tol=1e-6)
    res = amg_cg_solve_bind.amg_cg_solve_bind(levels, b, x0=x0, maxiter=100, tol=1e-6)
    print(res)
    # plot_residuals_all(allres,use_markers=True)


if __name__ == "__main__":
    demo_my_own_mg()