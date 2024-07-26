import ctypes
import scipy.sparse
import taichi as ti
import numpy as np
import time
import scipy
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
from pathlib import Path
import os,sys
from matplotlib import pyplot as plt
import shutil, glob
import meshio
import tqdm
import argparse
from collections import namedtuple
import json
import logging
import datetime
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial
from time import perf_counter
from scipy.linalg import pinv
import pyamg
import numpy.ctypeslib as ctl

sys.path.append(os.getcwd())

prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
cuda_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
os.add_dll_directory(cuda_dir)
extlib = ctl.load_library("fast-vcycle-gpu.dll", prj_path+'/cpp/mgcg_cuda/lib')

smoother_type = 'chebyshev'

arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
c_size_t = ctypes.c_size_t
c_float = ctypes.c_float
argtypes_of_csr=[ctl.ndpointer(np.float32,flags='aligned, c_contiguous'),    # data
                ctl.ndpointer(np.int32,  flags='aligned, c_contiguous'),      # indices
                ctl.ndpointer(np.int32,  flags='aligned, c_contiguous'),      # indptr
                ctypes.c_int, ctypes.c_int, ctypes.c_int           # rows, cols, nnz
                ]

chebyshev_coeff = None
def setup_chebyshev(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3,
                    iterations=1):
    global chebyshev_coeff # FIXME: later we should store this in the level
    """Set up Chebyshev."""
    rho = approximate_spectral_radius(A)
    a = rho * lower_bound
    b = rho * upper_bound
    # drop the constant coefficient
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]
    chebyshev_coeff = coefficients
    return coefficients

def chebyshev(A, x, b, coefficients=chebyshev_coeff, iterations=1):
    x = np.ravel(x)
    b = np.ravel(b)
    for _i in range(iterations):
        residual = b - A*x
        h = coefficients[0]*residual
        for c in coefficients[1:]:
            h = c*residual + A*h
        x += h

def setup_jacobi(A):
    from pyamg.relaxation.smoothing import rho_D_inv_A
    global jacobi_omega
    rho = rho_D_inv_A(A)
    print("rho:", rho)
    jacobi_omega = 1.0/(rho)
    print("omega:", jacobi_omega)


def build_Ps(A, method='UA'):
    """Build a list of prolongation matrices Ps from A """
    if method == 'UA':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None, symmetry='symmetric')
    elif method == 'SA' :
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400)
    elif method == 'CAMG':
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

def build_levels_cuda(A, Ps=[]):
    '''Give A and a list of prolongation matrices Ps, return a list of levels'''
    lvl = len(Ps) + 1 # number of levels

    levels = [MultiLevel() for i in range(lvl)]

    levels[0].A = A

    for i in range(lvl-1):
        levels[i].P = Ps[i]
    return levels


def setup_AMG(A):
    global chebyshev_coeff
    Ps = build_Ps(A)
    if smoother_type == 'chebyshev':
        setup_chebyshev(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)
    elif smoother_type == 'jacobi':
        setup_jacobi(A)
    return Ps


def old_amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    assert x0 is not None
    x = x0.copy()
    A = levels[0].A
    residuals = np.zeros(maxiter+1)
    def psolve(b):
        x = x0.copy()
        old_V_cycle(levels, 0, x, b)
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


def new_amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    tic1 = time.perf_counter()
    init_g_vcycle(levels)
    print(f"    init_g_vcycle time: {time.perf_counter()-tic1:.2f}s")
    assert g_vcycle

    tic2 = time.perf_counter()
    # set A0
    g_vcycle.fastmg_set_A0(levels[0].A.data.astype(np.float32), levels[0].A.indices, levels[0].A.indptr, levels[0].A.shape[0], levels[0].A.shape[1], levels[0].A.nnz)
    print(f"    set A0 time: {time.perf_counter()-tic2:.2f}s")
    
    # compute RAP   (R=P.T)
    tic3 = time.perf_counter()
    for lv in range(len(levels)-1):
        g_vcycle.fastmg_RAP(lv)
    print(f"    compute RAP time: {time.perf_counter()-tic3:.2f}s")

    if x0 is None:
        x0 = np.zeros(b.shape[0], dtype=np.float32)

    tic4 = time.perf_counter()
    # set data
    x0 = x0.astype(np.float32)
    b = b.astype(np.float32)
    g_vcycle.fastmg_set_mgcg_data(x0, x0.shape[0], b, b.shape[0], tol, maxiter)
    
    # solve
    g_vcycle.fastmg_mgcg_solve()

    # get result
    x = np.empty_like(x0, dtype=np.float32)
    residuals = np.empty(shape=(maxiter+1,), dtype=np.float32)
    niter = g_vcycle.fastmg_get_mgcg_data(x, residuals)
    residuals = residuals[:niter+1]
    print(f"    niter", niter)
    print(f"    solve time: {time.perf_counter()-tic4:.2f}s")
    return (x),  residuals  


def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag

def presmoother(A,x,b):
    from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
    if smoother_type == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif smoother_type == 'jacobi':
        jacobi(A,x,b,iterations=10, omega=jacobi_omega)
    elif smoother_type == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif smoother_type == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif smoother_type == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif smoother_type == 'chebyshev':
        chebyshev(A,x,b)


def postsmoother(A,x,b):
    presmoother(A,x,b)


def coarse_solver(A, b):
    res = np.linalg.solve(A.toarray(), b)
    return res

t_smoother = 0.0

def old_V_cycle(levels,lvl,x,b):
    global t_smoother
    A = levels[lvl].A.astype(np.float64)
    presmoother(A,x,b)
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        old_V_cycle(levels, lvl+1, coarse_x, coarse_b)
    x += levels[lvl].P @ coarse_x
    postsmoother(A, x, b)

g_vcycle = None
cached_P_id = None
cached_cheby_id = None
cached_jacobi_omega_id = None
def init_g_vcycle(levels):
    global g_vcycle
    global cached_P_id, cached_cheby_id, cached_jacobi_omega_id

    if g_vcycle is None:
        g_vcycle = extlib

        g_vcycle.fastmg_set_mgcg_data.argtypes = [arr_float, c_size_t, arr_float, c_size_t, c_float, c_size_t]
        g_vcycle.fastmg_get_mgcg_data.argtypes = [arr_float]*2
        g_vcycle.fastmg_get_mgcg_data.restype = c_size_t

        g_vcycle.fastmg_setup.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_setup_chebyshev.argtypes = [ctypes.c_size_t] * 2
        g_vcycle.fastmg_setup_jacobi.argtypes = [ctypes.c_float, ctypes.c_size_t]
        g_vcycle.fastmg_set_lv_csrmat.argtypes = [ctypes.c_size_t] * 11
        g_vcycle.fastmg_RAP.argtypes = [ctypes.c_size_t]

        g_vcycle.fastmg_set_A0.argtypes = argtypes_of_csr
        g_vcycle.fastmg_set_P.argtypes = [ctypes.c_size_t] + argtypes_of_csr

        g_vcycle.fastmg_setup(len(levels)) #just new fastmg instance and resize levels

    if smoother_type == 'chebyshev' and cached_cheby_id != id(chebyshev_coeff):
        cached_cheby_id = id(chebyshev_coeff)
        coeff_contig = np.ascontiguousarray(chebyshev_coeff, dtype=np.float32)
        g_vcycle.fastmg_setup_chebyshev(coeff_contig.ctypes.data, coeff_contig.shape[0])

    if smoother_type == 'jacobi' and cached_jacobi_omega_id != id(jacobi_omega):
        cached_jacobi_omega_id = id(jacobi_omega)
        g_vcycle.fastmg_setup_jacobi(jacobi_omega, 10)

    # set P
    if id(cached_P_id) != id(levels[0].P):
        cached_P_id = levels[0].P
        for lv in range(len(levels)-1):
            P_ = levels[lv].P
            g_vcycle.fastmg_set_P(lv, P_.data.astype(np.float32), P_.indices, P_.indptr, P_.shape[0], P_.shape[1], P_.nnz)


def main():
    from script.utils.define_to_read_dir import to_read_dir
    from script.utils.load_A_b import load_A_b

    A,b = load_A_b("F0-0")
    x0 = np.zeros(b.shape[0])

    global smoother_type
    smoother_type = 'jacobi'

    use_cuda = True
    if use_cuda:
        Ps = setup_AMG(A)
        levels = build_levels_cuda(A, Ps)
        x, r = new_amg_cg_solve(levels, b, x0=x0, tol=1e-6, maxiter=100)
    else:
        Ps = setup_AMG(A)
        levels = build_levels(A, Ps)
        x, r = old_amg_cg_solve(levels, b, x0=x0, tol=1e-6, maxiter=100)

    print("r", r)
    import matplotlib.pyplot as plt
    plt.plot(r)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()