import pyamg
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
# from pyamg.krylov import cg
from pyamg.relaxation.relaxation import gauss_seidel
from time import perf_counter
from scipy.linalg import pinv
# from scipy.sparse.linalg._isolve.utils import make_system


# 实现仅第一次进入coarse_solver时计算一次P
# https://stackoverflow.com/a/279597/19253199
def coarse_solver(A, b):
    if not hasattr(coarse_solver, "P"):
        coarse_solver.P = pinv(A.toarray())
    res = np.dot(coarse_solver.P, b)
    return res

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



# def cg11(A, b, x0=None, tol=1e-5, maxiter=None, M=None,residuals=None):
#     x = x0.copy()
#     # setup method
#     r = b - A @ x
#     z = M @ r
#     p = z.copy()
#     rz = np.inner(r.conjugate(), z)
#     normr = np.linalg.norm(r)
#     if residuals is not None:
#         residuals[:] = [normr]  # initial residual
#     # Check initial guess if b != 0,
#     normb = np.linalg.norm(b)
#     if normb == 0.0:
#         normb = 1.0  # reset so that tol is unscaled
#     # set the stopping criteria (see the docstring)
#     rtol = tol * normb
#     # How often should r be recomputed
#     recompute_r = 8
#     it = 0
#     while True:
#         Ap = A @ p
#         rz_old = rz
#         pAp = np.inner(Ap.conjugate(), p)         # check curvature of A
#         if pAp < 0.0:
#             # warn('\nIndefinite matrix detected in CG, aborting\n')
#             return (x, -1)
#         alpha = rz/pAp                            # 3
#         x += alpha * p                            # 4
#         if np.mod(it, recompute_r) and it > 0:    # 5
#             r -= alpha * Ap
#         else:
#             r = b - A @ x
#         z = M @ r                                 # 6
#         rz = np.inner(r.conjugate(), z)
#         if rz < 0.0:                             # check curvature of M
#             # warn('\nIndefinite preconditioner detected in CG, aborting\n')
#             return (x, -1)
#         beta = rz/rz_old                          # 7
#         p *= beta                                 # 8
#         p += z
#         it += 1
#         normr = np.linalg.norm(r)
#         if residuals is not None:
#             residuals.append(normr)
#         rtol = tol * normb
#         if normr < rtol:
#             return (x, 0)
#         if it == maxiter:
#             return (x, it)


# # preconditioned conjugate gradient
# # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
# # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
# # Note: Based on the scipy(https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/sparse/linalg/_isolve/iterative.py#L406), 
# # parameter M is actually the inverse of M in the wiki's formula. We adopt the scipy's definition.
# def my_pcg(A, b, x0, tol=1e-5, M=None, maxiter=100, residuals=[]):
#     x=x0.copy()
#     r=b-A@x
#     r=r.copy()
#     z = M@(r)
#     p=z.copy()
#     k=0
#     normb=np.linalg.norm(b)
#     while True:
#         Ap = A@p
#         rTz = r.T@z
#         alpha = r.T@z / (p.T@Ap)
#         x1 = x + alpha * p
#         r1 = r - alpha * Ap
#         normr = np.linalg.norm(r1)
#         residuals.append(normr)
#         if normr<tol*normb: 
#             break
#         if k>=maxiter:
#             break
#         z1 = M@(r1)
#         beta=r1.T@z1/(rTz)
#         p1=z1+beta*p
#         x=x1.copy()
#         r=r1.copy()
#         p=p1.copy()
#         z=z1.copy()
#         k+=1
#     return x1


# modified from scipy.sparse.linalg import cg
def cg22(A, b, x0=None, maxiter=None, M=None, rtol=1e-5, residuals=[]):
    x = x0.copy()   
    bnrm2 = np.linalg.norm(b)
    atol = rtol * bnrm2

    if bnrm2 == 0:
        return (b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    psolve = M.matvec
    r = b - A@(x) if x.any() else b.copy()

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return (x), 0

        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q = A@(p)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        residuals.append(np.linalg.norm(r))

    else:  # for loop exhausted
        # Return incomplete progress
        return (x), maxiter
    


def amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100,residuals=None, return_info=False):
    x = x0.copy()
    A = levels[0].A

    def matvec(b):
        x = x0.copy()
        V_cycle(levels, 0, x, b)
        return x
    
    M = LinearOperator(levels[0].A.shape, matvec, dtype=levels[0].A.dtype)

    tic = perf_counter()
    # x = my_pcg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, residuals=residuals)
    # x,info = cg11(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, residuals=residuals)
    x,info = cg22(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M, residuals=residuals)
    toc = perf_counter()

    print(f"amg_cg_solve Time: {toc-tic}")

    return x




# Only one iteration amg, for amg_cg to call. We dont need to check the residual,
# which will save some time.
def amg_standalone_solve_once(levels, b, x0):
    x = x0.copy()
    A = levels[0].A

    if len(levels) == 1:
        # hierarchy has only 1 level
        x = levels[-1].coarse_solver(A, b)
    else:
        V_cycle(levels, 0, x, b)
        
    toc = perf_counter()
    return x





def presmoother(A,x,b):
    gauss_seidel(A,x,b,iterations=1, sweep='symmetric')

def postsmoother(A,x,b):
    gauss_seidel(A,x,b,iterations=1, sweep='symmetric')


def V_cycle(levels,lvl,x,b):
    A = levels[lvl].A
    
    presmoother(A, x, b)
    
    residual = b - A @ x
    
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        V_cycle(levels, lvl+1, coarse_x, coarse_b)

    x += levels[lvl].P @ coarse_x

    postsmoother(A, x, b)



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

    residuals = []
    x0 = np.zeros_like(b)
    t2= perf_counter()
    x = amg_cg_solve(levels, b, x0=x0, maxiter=100, tol=1e-6, residuals=residuals)
    t3 = perf_counter()
    print('Solve Time:', t3-t2)
    # print('Total Time:', t3-t2+t1-t0)
    allres = [Residual('MyOwnMG', residuals, t3-t0)]

    tic = perf_counter()
    UA_CG(A, b, x0, allres)
    toc = perf_counter()
    print("UA_CG Time:", toc-tic)

    plot_residuals_all(allres,use_markers=True)


if __name__ == "__main__":
    demo_my_own_mg()