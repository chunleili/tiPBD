"""rep(means reproduced) 是可以复现pyamg的"""
import numpy as np
import scipy
from scipy.io import mmread, mmwrite
import scipy.sparse as sparse
import os, sys
from time import perf_counter
from matplotlib import pyplot as plt
import pyamg
from pyamg.gallery import poisson
from pyamg.relaxation.smoothing import change_smoothers
from collections import namedtuple
import argparse
from pathlib import Path

# from pyamg.relaxation import make_system
# from pyamg import amg_core

sys.path.append(os.getcwd())

prj_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("prj_dir", prj_dir)

parser = argparse.ArgumentParser()
parser.add_argument("-title", type=str, default=f"")
parser.add_argument("-f", type=int, default=20)
parser.add_argument("-case_name", type=str, default='test_A')
plot_title = parser.parse_args().title
frame = parser.parse_args().f
case_name = parser.parse_args().case_name

to_read_dir = prj_dir + f"result/{case_name}/A/"
save_fig = True
show_fig = True
generate_data = False
draw_plot = True
maxiter = 100
early_stop = False
tol=1e-10 # relative tolerance
run_concate_png = True
run_strength_options = False
postfix = ''


def test_amg(A, b, postfix=""):
    # x0 = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    allres = []

    run_amg_solvers(A,b,allres,x0)

    from script.utils.postprocess_residual import postprocess_allres
    df = postprocess_allres(allres)

    from script.utils.plot_residuals import plot_residuals_all
    plot_residuals_all(df, postfix=postfix)
    plt.show()

def run_amg_solvers(A,b,allres,x0):
    from script.utils.solvers import UA_CG_jacobi, diagCG, SA_from_diagnostic, UA_CG_jacobi, SA_CG, adaptive_SA_CG, CAMG_CG, amg_cuda_solvers, nullspace_UA_CG, amg_cuda_PCG
    # diagCG(A,b,x0,allres,tol=tol,maxiter=maxiter)
    # UA_CG_jacobi(A,b,x0,allres, tol=tol, maxiter=maxiter)
    # SA_CG(A,b,x0,allres, tol=tol, maxiter=maxiter)
    # CAMG_CG(A,b,x0,allres,tol=tol,maxiter=maxiter)
    # nullspace_UA_CG(A,b,x0,allres,tol=tol,maxiter=maxiter)
    # adaptive_SA_CG(A,b,x0,allres, tol=tol, maxiter=maxiter)
    amg_cuda_solvers(A,b,x0,allres,tol,maxiter,"nullspace","jacobi")
    amg_cuda_PCG(A,b,x0,allres,tol,maxiter)
    amg_cuda_solvers(A,b,x0,allres,tol,maxiter,"UA","jacobi")
    amg_cuda_solvers(A,b,x0,allres,tol,maxiter,"SA","jacobi")
    amg_cuda_solvers(A,b,x0,allres,tol,maxiter,"CAMG","jacobi")
    # amg_cuda_solvers(A,b,x0,allres,tol,maxiter,"adaptive_SA","jacobi")



def save_data(allres, postfix=""):
    import pandas as pd
    df = pd.DataFrame(allres)
    dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
    mkdir_if_not_exist(dir)
    df.to_csv(dir+f"/allres_{postfix}.csv")
    return df

def load_data(postfix=""):
    import pandas as pd
    dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
    df = pd.read_csv(dir+f"/allres_{postfix}.csv")
    return df



def prepare_A_b(mat_size = 10, case_num = 0, postfix=""):
    global A, b

    if(generate_data):
        print("generating data...")
        # A, b = generate_A_b_pyamg(n=mat_size)
        A, b = generate_A_b_spd(n=mat_size)
        scipy.io.mmwrite(to_read_dir + f"A{case_num}.mtx", A)
        np.savetxt(to_read_dir + f"b{case_num}.txt", b)
    else:
        from script.utils.load_A_b import  load_A_b
        A,b = load_A_b(postfix)
    return A,b

def improve_A(A):
    A = A + 1 * sparse.eye(A.shape[0])
    A = A.tocsr()
    return A

# def improve_A_make_M_matrix(A):
#     Anew = A.copy()
#     for i in range(Anew.shape[0]):
#         for j in range(Anew.shape[1]):
#             if i==j:
#                 continue
#             if Anew[i,j] > 0:
#                 Anew[i,j] = 0
#     return Anew

def improve_A_make_M_matrix(A):
    Anew = A.copy()
    diags = A.diagonal().copy()
    A.setdiag(np.zeros(A.shape[0]))
    A.data[A.data >0 ] = 0.0
    A.setdiag(diags)
    return Anew

def improve_A_by_remove_offdiag(A):
    A_downdiag = A.diagonal(-1)
    A_updiag = A.diagonal(1)
    A_diag = A.diagonal(0)
    newA = sparse.diags([A_downdiag, A_diag, A_updiag], [-1, 0, 1], format="csr")
    return newA

def improve_A_by_reduce_offdiag(A):
    A_diag = A.diagonal(0)
    A_diag_mat = sparse.diags([A_diag], [0], format="csr")
    A_offdiag = A - A_diag_mat
    A_offdiag = A_offdiag * 0.1
    newA = A_diag_mat + A_offdiag
    return newA

def improve_A_by_add_diag(A):
    diags = A.diagonal(0)
    diags += 1
    A.setdiag(diags)
    return A

def generate_R_P(A):
    print("generating R and P by pyamg...")
    # ml = pyamg.ruge_stuben_solver(A, max_levels=2)
    ml = pyamg.smoothed_aggregation_solver(A, max_levels=2)
    P = ml.levels[0].P
    R = ml.levels[0].R
    print(f"R: {R.shape}, P: {P.shape}")
    return R,P

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def timer_wrapper(func, *args, **kwargs):
    t = perf_counter()
    result = func(*args, **kwargs)
    print(f"{func.__name__} took {perf_counter() - t:.3e} s")
    return result


def generate_A_b_pyamg(n=10):
    # ---------------------- data generated by pyamg poisson --------------------- #
    A = poisson((n, n), format="csr")
    b = np.random.rand(A.shape[0])
    print(f"A: {A.shape}, b: {b.shape}")

    save = True
    if save:
        mmwrite("A.mtx", A)
        np.savetxt("b.txt", b)
    return A, b

def norm_two_grid_operator(A, R, P):
    # find spectral radius of I-S
    A2 = R @ A @ P
    A2_inv = scipy.sparse.linalg.inv(A2)
    S = P @ A2_inv @ R @ A
    I_S = np.identity(S.shape[0]) - S
    
    # norm of I_S
    # norm = A_norm(A, I_S)
    norm = np.linalg.norm(I_S)
    print("norm of two grid operator:", norm)
    return  norm

def spec_radius_two_grid_operator(A, R, P):
    # find spectral radius of I-S
    A2 = R @ A @ P
    A2_inv = scipy.sparse.linalg.inv(A2)
    S = P @ A2_inv @ R @ A

    I_S = np.identity(S.shape[0]) - S
    eigens = scipy.sparse.linalg.eigs(I_S)
    spec_radius = max(abs(eigens[0]))
    print("eigens:", eigens[0])
    print("spec_radius:", spec_radius)
    return spec_radius

# judge if A is positive definite
# https://stackoverflow.com/a/44287862/19253199
# if A is symmetric and able to be Cholesky decomposed, then A is positive definite
def is_spd(A):
    A=A.toarray()
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            print("A is positive definite")
            return True
        except np.linalg.LinAlgError:
            print("A is not positive definite")
            return False
    else:
        print("A is not positive definite")
        return False

def generate_A_b_spd(n=1000):
    import scipy.sparse as sp
    A = sp.random(n, n, density=0.01, format="csr")
    A = A.T @ A
    b = np.random.rand(A.shape[0])
    flag = is_spd(A)
    print(f"is_spd: {flag}")
    print(f"Generated spd A: {A.shape}, b: {b.shape}")
    A = sp.csr_matrix(A)
    return A, b


def print_residuals(residuals, name="residuals"):
    for i, r in enumerate(residuals):
        print(f"{name}[{i}] = {r:.8e}")


def analyse_A(A,R,P):
    spec_radius_two_grid_operator(A, R, P)
    norm_TG = norm_two_grid_operator(A, R, P)
    print("A norm of TG:", norm_TG)
    codition_number_of_A = np.linalg.cond(A.toarray())
    print(f"condition number of A: {codition_number_of_A}")
    print("A is symmetric:", np.array_equal(A.toarray(), A.toarray().T))
    print("singular values of P:", np.linalg.svd(P.toarray())[1])
    rank_P = np.linalg.matrix_rank(P.toarray())
    print("rank of P:", rank_P)
    eigenvalues_A = np.linalg.eigvals(A.toarray())
    print("eigenvalues of A:", eigenvalues_A)
    print("R@P is:", R@P)

def solve_pyamg(ml, b):
    residuals = []
    x = ml.solve(b, tol=1e-3, residuals=residuals, maxiter=maxiter)
    return x, residuals

def solve_FAS(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
    maxiter = 1

    A2 = R @ A @ P
    x0 = np.zeros_like(b) # FIXME in the future, x0 should be a parameter
    x = x0.copy()

    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0

    while True:  # it <= maxiter and normr >= tol:
        # gauss_seidel(A, x, b, iterations=1)  # presmoother
        residual = b - A @ x
        v_c = R@x
        coarse_b = R @ residual + A2@v_c  # restriction
        coarse_x = scipy.sparse.linalg.spsolve(A2, coarse_b)
        x += P @ coarse_x  # coarse grid correction
        # gauss_seidel(A, x, b, iterations=1)  # postsmoother
        it += 1
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x
        

def solve_rep_noSmoother(A, b, x0, R, P):
    residuals=[]
    tol = 1e-3
    maxiter = 1
    x0 = np.zeros_like(b) # FIXME in the future, x0 should be a parameter

    A2 = R @ A @ P

    x = x0.copy()

    # normb = np.linalg.norm(b)
    normb = A_norm(A, b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    # normr = np.linalg.norm(b - A @ x)
    normr = A_norm(A, b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0
    while True:  # it <= maxiter and normr >= tol:
        # gauss_seidel(A, x, b, iterations=1)  # presmoother

        residual = b - A @ x

        coarse_b = R @ residual  # restriction

        coarse_x = np.zeros_like(coarse_b)

        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)

        x += P @ coarse_x  # coarse grid correction

        # gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        # normr = np.linalg.norm(b - A @ x)
        normr = A_norm(A, b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x, residuals
        if it == maxiter:
            return x, residuals


def solve_rep(A, b, x0, R, P, maxiter=1, tol=1e-6):
    residuals = []
    full_residual = [[],[],[],[]]

    A2 = R @ A @ P
    x0 = np.zeros_like(b) # initial guess x0
    x = x0.copy()

    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual
    full_residual[0] = (b - A @ x)

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0
    while True:  # it <= maxiter and normr >= tol:
        gauss_seidel(A, x, b, iterations=1)  # presmoother

        residual = b - A @ x
        full_residual[1] = residual

        coarse_b = R @ residual  # restriction

        coarse_x = np.zeros_like(coarse_b)

        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)

        dx = P @ coarse_x  # coarse grid correction
        x += dx  # coarse grid correction

        full_residual[2] = b - A @ x

        gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        full_residual[3] = (b - A @ x)
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x, residuals, full_residual
        if it == maxiter:
            return x, residuals, full_residual



def solve_onlySmoother(A, b, x0, R, P, maxiter=1, tol=1e-6):
    residuals = []

    # A2 = R @ A @ P
    x0 = np.zeros_like(b) # initial guess x0
    x = x0.copy()

    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0
    while True:  # it <= maxiter and normr >= tol:
        gauss_seidel(A, x, b, iterations=1)  # presmoother

        residual = b - A @ x

        coarse_b = R @ residual  # restriction

        coarse_x = np.zeros_like(coarse_b)

        # coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)

        x += P @ coarse_x  # coarse grid correction

        gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x, residuals
        if it == maxiter:
            return x, residuals




def solve_amg(A, b, x0, R, P, residuals=[], maxiter = 1, tol = 1e-6):
    A2 = R @ A @ P
    x = x0.copy()
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual
    b = np.ravel(b)
    x = np.ravel(x)
    it = 0
    while True:  # it <= maxiter and normr >= tol:
        gauss_seidel(A, x, b, iterations=1)  # presmoother
        residual = b - A @ x
        coarse_b = R @ residual  # restriction
        coarse_x = np.zeros_like(coarse_b)
        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)
        x += P @ coarse_x 
        gauss_seidel(A, x, b, iterations=1)
        it += 1
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x

def solve_rep_Anorm(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
    maxiter = 1

    A2 = R @ A @ P
    x0 = np.zeros_like(b) # FIXME in the future, x0 should be a parameter
    x = x0.copy()

    normb = A_norm(A, b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    # normr = np.linalg.norm(b - A @ x)
    normr = A_norm(A, b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0
    while True:  # it <= maxiter and normr >= tol:
        gauss_seidel(A, x, b, iterations=1)  # presmoother

        residual = b - A @ x

        coarse_b = R @ residual  # restriction

        coarse_x = np.zeros_like(coarse_b)

        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)

        x += P @ coarse_x  # coarse grid correction

        gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        # normr = np.linalg.norm(b - A @ x)
        normr = A_norm(A, b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x

def A_norm(A,x):
    '''
    A-norm = x^T A x
    '''
    return x.T @ A @ x


def gauss_seidel(A, x, b, iterations=1, residuals = [], tol=1e-6):
    # if not scipy.sparse.isspmatrix_csr(A):
    #     raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        # forward sweep
        for _ in range(1):
            amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

        # backward sweep
        for _ in range(1):
            amg_core_gauss_seidel_kernel(
                A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
            )
        
        normr = np.linalg.norm(b - A @ x)
        residuals.append(normr)

        if early_stop:
            if normr < tol:
                break
    return x


def amg_core_gauss_seidel(Ap, Aj, Ax, x, b, row_start: int, row_stop: int, row_step: int):
    for i in range(row_start, row_stop, row_step):
        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag


import taichi as ti
ti.init()

@ti.kernel
def amg_core_gauss_seidel_kernel(Ap: ti.types.ndarray(),
                                 Aj: ti.types.ndarray(),
                                 Ax: ti.types.ndarray(),
                                 x: ti.types.ndarray(),
                                 b: ti.types.ndarray(),
                                 row_start: int,
                                 row_stop: int,
                                 row_step: int):
    # if row_step < 0:
    #     assert "row_step must be positive"
    for i in range(row_start, row_stop):
        if i%row_step != 0:
            continue

        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag



def solve_simplest(A, b, R, P, residuals):
    tol = 1e-3
    maxiter = 1
    A2 = R @ A @ P
    x0 = np.zeros_like(b) # initial guess x0
    x = x0.copy()
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual
    b = np.ravel(b)
    x = np.ravel(x)
    it = 0
    while True:  # it <= maxiter and normr >= tol:
        residual = b - A @ x
        gauss_seidel(A,x,b) # pre smoother
        coarse_b = R @ residual  # restriction
        coarse_x = np.zeros_like(coarse_b)
        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)
        x += P @ coarse_x 
        # amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
        gauss_seidel(A, x, b) # post smoother
        it += 1
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x


def strength_options(A,b):
    import numpy as np
    import pyamg
    import matplotlib.pyplot as plt
    import time

    # n = int(1e2)
    # stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=np.pi / 3)
    # A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
    # b = np.random.rand(A.shape[0])
    # A,b = prepare_A_b(case_name)
    x0 = 0 * b

    runs = []
    options = []
    options.append(('symmetric', {'theta': 0.0}))
    options.append(('symmetric', {'theta': 0.25}))
    options.append(('evolution', {'epsilon': 4.0}))
    options.append(('affinity', {'epsilon': 3.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    options.append(('affinity', {'epsilon': 4.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    options.append(('algebraic_distance',
                {'epsilon': 2.0, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
    options.append(('algebraic_distance',
                {'epsilon': 3.0, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))

    for opt in options:
        #optstr = opt[0] + '\n    ' + \
        #    ',\n    '.join(['%s=%s' % (u, v) for (u, v) in list(opt[1].items())])
        optstr = opt[0] + ': ' + \
            ', '.join(['%s=%s' % (u, v) for (u, v) in list(opt[1].items())])
        print("running %s" % (optstr))

        tic = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength=opt,
            max_levels=15,
            max_coarse=300,
            keep=False)
        res = []
        x = ml.solve(b, x0, tol=1e-12, residuals=res)
        runs.append((res, optstr))
        print(f"Elapsed time: {time.perf_counter() - tic:0.4f} seconds")

    fig, ax = plt.subplots()
    for run in runs:
        label = run[1]
        label = label.replace('theta', '$\\theta$')
        label = label.replace('epsilon', '$\\epsilon$')
        label = label.replace('alpha', '$\\alpha$')
        ax.semilogy(run[0], label=label, linewidth=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual')

    #l4 = plt.legend(bbox_to_anchor=(0,1.02,1,0.5), loc="lower left",
    #                mode="expand", borderaxespad=0, ncol=1)
    plt.legend(loc="lower left", borderaxespad=0, ncol=1, frameon=False)
    plt.title(f'{case_name}: Strength Options')

    # figname = f'./output/strength_options.png'
    import sys
    # if '--savefig' in sys.argv:
    if save_fig:
        plt.savefig(prj_dir+f"/result/{case_name}/png/strength_{plot_title}.png")
    if show_fig:
        plt.show()

def plot_residuals(data, ax, *args, **kwargs):
    title = kwargs.pop("title", "")
    linestyle = kwargs.pop("linestyle", "-")
    label = kwargs.pop("label", "")
    x = np.arange(len(data))
    ax.plot(x, data, label=label, linestyle=linestyle, *args, **kwargs)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.legend(loc="upper right")


def mkdir_if_not_exist(path=None):
    from pathlib import Path
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)

def test_different_N():
    global plot_title, generate_data
    generate_data = True
    for case_num in range(100):
        N = np.random.randint(100, 5000)
        plot_title = f"case_{case_num}_A_size_{N}"
        print(f"\ncase:{case_num}\tN: {N}")
        test_amg(N, case_num)

def test_all_A():
    for frame in range(30,100,10):
        for ite in range(0,50,30):
            postfix = f"F{frame}_{ite}"
            global plot_title
            plot_title = postfix
            test_amg(10, 0, postfix)



def test_6():
    for i in range(1,30,5):
        postfix = f"F{i}-0"
        plot_title =  postfix   
        print(f"{postfix}")
        A,b = prepare_A_b(postfix=postfix)
        if run_strength_options:
            strength_options(A,b)
        else:
            test_amg(A,b,postfix)

    if run_concate_png:
        import script.utils.concatenate_png as concatenate_png
        if run_strength_options : prefix = 'strength'
        else: prefix = 'residuals'
        concatenate_png.concatenate_png(case_name, prefix)



def draw_saved_data(postfix="F1"):
    allres = load_data(postfix)
    from script.utils.plot_residuals import plot_residuals_all_new
    plot_residuals_all_new(allres)


def generate_data_from_sim():
    import subprocess,os
    # go to the root dir of the project

    print("generating data...")
    mu = 1e7
    dt = 15e-3
    # for mu in [1e6, 1e7, 1e8]:
    #     for dt in [1e-3, 1e-4]:
    args = ["python",
            "engine/soft/soft3d.py",
            f"-out_dir=result/test_A",
            "-model_path=data/model/bunny_small/bunny_small.node",
            f"-tol=1e-4",
            f"-delta_t={dt}",
            "-solver_type=AMG",
            "-arch=cpu",
            "-maxiter=2",
            "-smoother_niter=2",
            "-build_P_method=strength0.1",
            "-end_frame=1",
            f"-export_matrix=1",
            f"-mu={mu}"
            ]
    subprocess.check_call(args)
        


if __name__ == "__main__":
    # draw_saved_data()
    generate_data_from_sim()

    print("first run python run.py -A soft  to generate data!")

    for postfix in ["F1"]:
        print(f"\n\n\n{postfix}")
        A,b = prepare_A_b(postfix=postfix)
        test_amg(A,b,postfix=postfix)

    # import script.utils.concatenate_png as concatenate_png
    # concatenate_png.concatenate_png(case_name, prefix='residuals', frames=frames)
