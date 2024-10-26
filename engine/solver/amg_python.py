import numpy as np
import time
from time import perf_counter
import logging
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients

from engine.solver.build_Ps import build_Ps


def AMG_python(b, args, ist, fill_A_csr_ti, should_setup, copy_A=True):
    A = fill_A_csr_ti(ist)
    if copy_A:
        A = A.copy()#FIXME: softbody no copy will cause bug, but cloth is good, why?

    if should_setup():
        tic = time.perf_counter()
        ist.Ps = build_Ps(A, args, ist)
        ist.num_levels = len(ist.Ps)+1
        logging.info(f"    build_Ps time:{time.perf_counter()-tic}")
    
    tic = time.perf_counter()
    levels = build_levels(A, ist.Ps)
    logging.info(f"    build_levels time:{time.perf_counter()-tic}")

    if should_setup():
        tic = time.perf_counter()
        setup_smoothers(A, args, ist)
        logging.info(f"    setup smoothers time:{perf_counter()-tic}")
    x0 = np.zeros_like(b)
    tic = time.perf_counter()
    x, r_Axb = old_amg_cg_solve(args,ist,levels, b, x0=x0, maxiter=args.maxiter_Axb, tol=1e-6)
    toc = time.perf_counter()
    logging.info(f"    mgsolve time {toc-tic}")
    return  x, r_Axb


# https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/relaxation/relaxation.py#L586
def chebyshev(A, x, b, ist):
    coefficients = ist.chebyshev_coeff
    iterations = 1
    x = np.ravel(x)
    b = np.ravel(b)
    for _i in range(iterations):
        residual = b - A*x
        h = coefficients[0]*residual
        for c in coefficients[1:]:
            h = c*residual + A*h
        x += h

def calc_spectral_radius(A,ist):
    t = time.perf_counter()
    ist.spectral_radius = approximate_spectral_radius(A) # legacy python version
    print(f"spectral_radius time: {time.perf_counter()-t:.2f}s")
    print("spectral_radius:", ist.spectral_radius)
    return ist.spectral_radius


def setup_chebyshev(A, ist):
    """Set up Chebyshev."""
    lower_bound=1.0/30.0
    upper_bound=1.1
    degree=3
    rho = calc_spectral_radius(A,ist)
    a = rho * lower_bound
    b = rho * upper_bound
    ist.chebyshev_coeff = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]


def setup_jacobi(A,ist):
    from pyamg.relaxation.smoothing import rho_D_inv_A
    rho = rho_D_inv_A(A)
    print("rho:", rho)
    ist.jacobi_omega = 1.0/(rho)
    print("omega:", ist.jacobi_omega)


def setup_smoothers(A,args,ist):
    if args.smoother_type == 'chebyshev':
        setup_chebyshev(A, ist)
    elif args.smoother_type == 'jacobi':
        setup_jacobi(A,ist)


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


def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag

def presmoother(A,x,b,args,ist):
    from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
    if args.smoother_type == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif args.smoother_type == 'jacobi':
        jacobi(A,x,b,iterations=10, omega=ist.jacobi_omega)
    elif args.smoother_type == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif args.smoother_type == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif args.smoother_type == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif args.smoother_type == 'chebyshev':
        chebyshev(A,x,b,ist)


def postsmoother(A,x,b,args,ist):
    presmoother(A,x,b,args,ist)


def coarse_solver(A, b):
    res = np.linalg.solve(A.toarray(), b)
    return res

def old_V_cycle(levels,lvl,x,b,args,ist):
    A = levels[lvl].A.astype(np.float64)
    presmoother(A,x,b,args,ist)
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        old_V_cycle(levels, lvl+1, coarse_x, coarse_b,args,ist)
    x += levels[lvl].P @ coarse_x
    postsmoother(A, x, b,args,ist)



def old_amg_cg_solve(args, ist, levels, b, x0=None, tol=1e-5, maxiter=100):
    assert x0 is not None
    x = x0.copy()
    A = levels[0].A
    residuals = np.zeros(maxiter+1)
    def psolve(b):
        x = x0.copy()
        old_V_cycle(levels, 0, x, b, args, ist)
        return x
    bnrm2 = np.linalg.norm(b)
    atol = tol * bnrm2
    r = b - A@(x)
    rho_prev, p = None, None
    normr = np.linalg.norm(r)
    residuals[0] = normr
    iteration = 0
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

