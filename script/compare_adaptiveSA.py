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

def setup_chebyshev(lvl, lower_bound=1.0/30.0, upper_bound=1.1, degree=3,
                    iterations=1):
    global chebyshev_coeff # FIXME: later we should store this in the level
    """Set up Chebyshev."""
    rho = approximate_spectral_radius(lvl.A)
    a = rho * lower_bound
    b = rho * upper_bound
    # drop the constant coefficient
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]
    chebyshev_coeff = coefficients
    return coefficients


def chebyshev(A, x, b):
    polynomial(A, x, b, coefficients=chebyshev_coeff, iterations=1)


def build_Ps(A, method='UA_NoImprove'):
    """Build a list of prolongation matrices Ps from A """
    if method == 'UA' or method == 'UA_CG':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None)
    elif method == 'SA' or method == 'SA_CG':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400)
    elif method == 'UA_CG_GS':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, coarse_solver='gauss_seidel')
    elif method == 'CAMG' or method == 'CAMG_CG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400)
    elif method == 'UA_NoImprove':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None)
    elif method == 'adaptive_SA':
        ml = pyamg.aggregation.adaptive_sa_solver(A, max_coarse=400, smooth=None, num_candidates=3)[0]
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


# 实现仅第一次进入coarse_solver时计算一次P
# https://stackoverflow.com/a/279597/19253199
def coarse_solver(A, b):
    global update_coarse_solver
    if not hasattr(coarse_solver, "P") or update_coarse_solver:
        coarse_solver.P = pinv(A.toarray())
        update_coarse_solver = False
    res = np.dot(coarse_solver.P, b)
    return res

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


# def V_cycle_norecur(levels,lvl,x,b):
#     lvl = 0
#     x0=x
#     b0=b
#     A0 = levels[lvl].A
#     gauss_seidel(A0,x0,b0,iterations=1, sweep='symmetric') # presmoother
#     residual0 = b0 - A0 @ x0
#     b1 = levels[lvl].R @ residual0

#     lvl = 1
#     x1 = np.zeros_like(b1)
#     A1 = levels[lvl].A
#     gauss_seidel(A1,x1,b1,iterations=1, sweep='symmetric') # presmoother
#     residual1 = b1 - A1 @ x1
#     b2 = levels[lvl].R @ residual1

#     lvl = 2
#     A2 = levels[lvl].A
#     x2 = coarse_solver(A2, b2)

#     lvl = 1
#     x1 += levels[lvl].P @ x2
#     gauss_seidel(A1,x1,b1,iterations=1, sweep='symmetric') # postsmoother

#     lvl = 0
#     x0 += levels[lvl].P @ x1
#     gauss_seidel(A0,x0,b0,iterations=1, sweep='symmetric') # postsmoother

#     x = x0



# def V_cycle_norecur_new(levels,lvl,x,b):
#     levels[0].x = x
#     levels[0].b = b

#     for l in range(len(levels)-1):
#         levels[l].x = np.zeros(levels[l].A.shape[0])
#         gauss_seidel(levels[l].A, levels[l].x , levels[l].b,  iterations=1, sweep='symmetric')
#         levels[l+1].b = levels[l].R @ (b - levels[l].A @ x)

#     l = len(levels)-1 # coarsest level: 2
#     x = coarse_solver(levels[l].A, levels[l].b)

#     for l in range(len(levels)-2, -1, -1): # 1, 0
#         levels[l].x += levels[l].P @ levels[l+1].x
#         gauss_seidel(levels[l].A, levels[l].x , b,  iterations=1, sweep='symmetric')

#     x = levels[0].x


# def unpack_levels(levels):
#     global lmax, xs, bs, As, Rs, Ps
#     lmax = len(levels)-1 # 2
#     numl = len(levels)
#     xs = [None] * (numl)
#     bs = [None] * (numl)
#     As = [None] * (numl)
#     Rs = [None] * (numl-1)
#     Ps = [None] * (numl-1)
#     for i in range(numl):
#         xs[i] = np.zeros(levels[i].A.shape[0])
#         bs[i] = np.zeros(levels[i].A.shape[0])
#         As[i] = levels[i].A
#     for i in range(numl-1):
#         Rs[i] = levels[i].R
#         Ps[i] = levels[i].P


# def V_cycle_norecur_newnew(b):
#     bs[0] = b
#     for i in range(lmax-1): # 0, 1
#         presmoother(As[i], xs[i] , bs[i])
#         bs[i+1] = Rs[i] @ (bs[i] - As[i] @ xs[i])
#     xs[lmax] = coarse_solver(As[lmax], bs[lmax])
#     for i in range(lmax-1, -1, -1): # 1, 0
#         xs[i] += Ps[i] @ xs[i+1]
#         postsmoother(As[i], xs[i] , bs[i])
#     return xs[0]


def setup_AMG(A):
    Ps = build_Ps(A)
    levels = build_levels(A, Ps)
    chebyshev = setup_chebyshev(levels[0], lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)
    return levels, chebyshev


def AMGCG(A,b):
    # for _ in range(5):
    t0= perf_counter()
    Ps = build_Ps(A)
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    print('Setup Time:', t1-t0)

    chebyshev = setup_chebyshev(levels[0], lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)

    smoother = 'chebyshev'
    tic = perf_counter()
    x0 = np.zeros_like(b)
    x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=100, tol=1e-6)
    toc = perf_counter()
    print("My chebyshev Time:", toc-tic)
    # allres.append(Residual('MyChebyshev', residuals, toc-tic))



def compare_adaptive_SA(postfix='F10-0'):
    import os, sys
    sys.path.append(os.getcwd())
    from utils.load_A_b import load_A_b
    from utils.solvers import UA_CG, UA_CG_chebyshev, UA_CG_jacobi, CG
    from collections import namedtuple
    from utils.plot_residuals import plot_residuals_all
    from utils.postprocess_residual import print_allres_time, calc_conv
    from utils.parms import maxiter
    Residual = namedtuple('Residual', ['label','r', 't'])
    global smoother, chebyshev, levels


    A, b = load_A_b(postfix)

    allres = []

    # UA
    t0= perf_counter()
    Ps = build_Ps(A, method='UA_NoImprove')
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    setup_chebyshev(levels[0], lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)
    print('Setup Time:', t1-t0)
    print(f"levels:{len(levels)}")
    for i in range(len(levels)):
        print(f"level {i} shape: {levels[i].A.shape}")

    # 注意！chebyshev对软体是不收敛的!
    smoother = 'gauss_seidel'
    tic = perf_counter()
    x0 = np.zeros_like(b)
    x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=maxiter, tol=1e-6)
    toc = perf_counter()
    print("solve phase:", toc-tic)
    allres.append(Residual('UA', residuals, toc-tic))
    conv = calc_conv(residuals)
    print("conv:", conv)






    # adaptive SA
    print("===========adaptive SA==========")
    global update_coarse_solver
    update_coarse_solver = True
    t0= perf_counter()
    Ps = build_Ps(A,method='adaptive_SA')
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    setup_chebyshev(levels[0], lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)
    print('Setup Time:', t1-t0)
    print(f"levels:{len(levels)}")
    for i in range(len(levels)):
        print(f"level {i} shape: {levels[i].A.shape}")

    # smoother = 'chebyshev'
    smoother = 'gauss_seidel'
    tic = perf_counter()
    x0 = np.zeros_like(b)
    x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=maxiter, tol=1e-6)
    toc = perf_counter()
    allres.append(Residual('adaptive_SA', residuals, toc-tic))
    print("solve phase:", toc-tic)
    conv = calc_conv(residuals)
    print("conv:", conv)

    print_allres_time(allres, draw=True)
    plot_residuals_all(allres,use_markers=True)

    # adpative 对软体效果比UA好，但是布料上几乎无差别。

if __name__ == "__main__":
    compare_adaptive_SA()
