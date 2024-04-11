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

# from pyamg.relaxation import make_system
# from pyamg import amg_core

sys.path.append(os.getcwd())

prj_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("prj_dir", prj_dir)
to_read_dir = prj_dir + "result/test/"

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=100)
N = parser.parse_args().N
print(f"N={N}")
parser.add_argument("-title", type=str, default=f"")
plot_title = parser.parse_args().title
parser.add_argument("-f", type=int, default=10)
frame = parser.parse_args().f
save_fig_instad_of_show = False
generate_data = False
show_plot = True

def test_amg(mat_size = 10):
    # ------------------------------- prepare data ------------------------------- #
    if(generate_data):
        print("generating data...")
        # A, b = generate_A_b_pyamg(n=mat_size)
        A, b = generate_A_b_psd(n=mat_size)
    else:
        print("loading data...")
        A = scipy.io.mmread(to_read_dir+f"A.mtx")
        A = A.tocsr()
        # b = np.loadtxt(to_read_dir+f"b.txt", dtype=np.float32)
        b= np.ones(A.shape[0])

    # generate R by pyamg
    print("generating R and P by pyamg...")

    # ml = pyamg.ruge_stuben_solver(A, max_levels=2)
    ml = pyamg.smoothed_aggregation_solver(A, max_levels=2)
    P = ml.levels[0].P
    R = ml.levels[0].R
    print(f"R: {R.shape}, P: {P.shape}")

    # spec_radius_two_grid_operator(A, R, P)
    norm_TG = norm_two_grid_operator(A, R, P)
    print("A norm of TG:", norm_TG)
    codition_number_of_A = np.linalg.cond(A.toarray())
    print(f"condition number of A: {codition_number_of_A}")
    # judege symmetric:
    # print("A is symmetric:", np.array_equal(A.toarray(), A.toarray().T))
    # print("singular values of P:", np.linalg.svd(P.toarray())[1])
    rank_P = np.linalg.matrix_rank(P.toarray())
    print("rank of P:", rank_P)
    eigenvalues_A = np.linalg.eigvals(A.toarray())
    print("eigenvalues of A:", eigenvalues_A)
    # print("R@P is:", R@P)

    # ------------------------------- test solvers ------------------------------- #
    x0 = np.zeros_like(b)

    # print("Solving pyamg...")
    # residuals_pyamg = []
    # _,residuals_pyamg = timer_wrapper(solve_pyamg, ml, b)

    # print("Solving rep...")
    # x0 = np.zeros_like(b)
    # _,residuals_rep = timer_wrapper(solve_rep, A, b, x0, R, P)

    print("Solving rep_noSmoother...")
    residuals_noSmoother = []
    x0 = np.zeros_like(b)
    _,residuals_noSmoother = timer_wrapper(solve_rep_noSmoother, A, b, x0, R, P)

    print("generating R and P by selecting row...")
    R2 = scipy.sparse.csr_matrix((2,A.shape[0]), dtype=np.int32)
    R2[0,0] = 1
    R2[1,9] = 1
    P2 = R2.T
    x0 = np.zeros_like(b)
    _,residuals_selectRows = timer_wrapper(solve_rep_noSmoother, A, b, x0, R2, P2)

    print("generating R and P by removing rows...")
    R3 = scipy.sparse.identity(A.shape[0], dtype=np.int32)
    R3=R3.tocsr()
    R3 = delete_rows_csr(R3, range(0, A.shape[0] - 1, 2))
    P3 = R3.T
    print(f"##########R: {R3.shape}, P: {P3.shape}")
    x0 = np.zeros_like(b)
    print("rank of P3:", np.linalg.matrix_rank(P3.toarray()))
    _,residuals_removeRows = timer_wrapper(solve_rep_noSmoother, A, b, x0, R3, P3)

    # ------------------------------- print results ---------------------------- #
    print_residuals(residuals_noSmoother, "rep_noSmoother")
    print_residuals(residuals_selectRows, "selectRows")
    print_residuals(residuals_removeRows, "removeRows")

    if show_plot:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        plot_residuals(residuals_noSmoother, axs, title=plot_title, linestyle="--",label="pyamg noSmoother")
        plot_residuals(residuals_selectRows, axs, title=plot_title, linestyle="--",label="SelectRow")
        plot_residuals(residuals_removeRows, axs, title=plot_title, linestyle="--",label="removeRows")

        fig.canvas.manager.set_window_title(plot_title)
        plt.tight_layout()
        if save_fig_instad_of_show:
            plt.savefig(f"result/test/residuals_{plot_title}.png")
        else:
            plt.show()


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
def is_pos_def(A):
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
        
def generate_A_b_psd(n=1000):
    # ---------------------- data generated by pyamg poisson --------------------- #
    # A = np.random.rand(n, n)
    A = sparse.random(n, n, density=0.01, format="csr")
    A = A.T @ A
    b = np.random.rand(A.shape[0])
    # is_pos_def(A)
    print(f"Generated PSD A: {A.shape}, b: {b.shape}")
    A = sparse.csr_matrix(A)
    return A, b

def print_residuals(residuals, name="residuals"):
    for i, r in enumerate(residuals):
        print(f"{name}[{i}] = {r:.8e}")



def solve_pyamg(ml, b):
    residuals = []
    x = ml.solve(b, tol=1e-3, residuals=residuals, maxiter=1)
    return x, residuals

def solve_FAS(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
    maxiter = 1

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


def solve_rep(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
    maxiter = 1

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

        x += P @ coarse_x  # coarse grid correction

        gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x

def solve_amg(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
    maxiter = 1
    A2 = R @ A @ P
    x = x0
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

def A_norm2(A,x):
    norm = (x.T @ A @ x)
    res = 
    return  

def gauss_seidel(A, x, b, iterations=1):
    if not sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        # forward sweep
        # print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

        # backward sweep
        # print("backward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(
                A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
            )
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

def plot_residuals(data, ax, *args, **kwargs):
    title = kwargs.pop("title", "")
    linestyle = kwargs.pop("linestyle", "-")
    label = kwargs.pop("label", "")
    x = np.arange(len(data))
    ax.plot(x, data, label=label, linestyle=linestyle, *args, **kwargs)
    ax.set_title(title)
    # ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.legend(loc="upper right")



def test_different_N():
    global plot_title
    for case_num in range(100):
        N = np.random.randint(1000, 20000)
        plot_title = f"case_{case_num}_A_size_{N}"
        print(f"\ncase:{case_num}\tN: {N}")
        test_amg(N)

if __name__ == "__main__":
    test_amg(1000)
    # test_amg(20)
    # test_amg(30)
    # test_amg(50)
    # test_amg(100)
