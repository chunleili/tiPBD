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
        b = np.loadtxt(to_read_dir+f"b.txt", dtype=np.float32)

    # generate R by pyamg
    print("generating R and P by pyamg...")
    ml = pyamg.ruge_stuben_solver(A, max_levels=2)
    P = ml.levels[0].P
    R = ml.levels[0].R
    # R = mmread("R.mtx")
    # P = mmread("P.mtx")
    print(f"R: {R.shape}, P: {P.shape}")

    # spec_radius_two_grid_operator(A, R, P)
    # codition_number_of_A = np.linalg.cond(A.toarray())
    # print(f"condition number of A: {codition_number_of_A}")
    # judege symmetric:
    print("A is symmetric:", np.array_equal(A.toarray(), A.toarray().T))

    # ------------------------------- test solvers ------------------------------- #
    x0 = np.zeros_like(b)

    print("Solving pyamg...")
    r_norms_pyamg = []
    x_pyamg = timer_wrapper(solve_pyamg, ml, b, r_norms_pyamg)

    print("Solving rep...")
    r_norms_rep = []
    x_rep = timer_wrapper(solve_rep, A, b, x0, R, P, r_norms_rep)

    print("Solving rep_noSmoother...")
    r_norms_noSmoother = []
    x_noSmoother, coarse_residuals = timer_wrapper(solve_rep_noSmoother, A, b, x0, R, P, r_norms_noSmoother)

    print("Solving FAS...")
    r_norms_FAS = []
    x_FAS = timer_wrapper(solve_FAS, A, b, x0, R, P, r_norms_FAS)

    print("Solving AMG...")
    r_norms_AMG = []
    x_AMG = timer_wrapper(solve_amg, A, b, x0, R, P, r_norms_AMG)

    # print("Solving simplest...")
    # r_norms_simplest = []
    # x_simplest = timer_wrapper(solve_simplest, A, b, R, P, r_norms_simplest)

    # print("Solving rep_Anorm...")
    # r_norms_repAnorm = []
    # x_rep = timer_wrapper(solve_rep_Anorm, A, b, x0, R, P, r_norms_repAnorm)


    # print("Solving by direct solver...")
    # r_norms_direct = []
    # r_norms_direct.append(np.linalg.norm(b))
    # x_direct = scipy.sparse.linalg.spsolve(A, b)
    # r_norms_direct.append(np.linalg.norm(b - A @ x_direct))

    # print("Solving by GS")
    # r_norms_GS = []
    # r_norms_GS.append(np.linalg.norm(b))
    # x_GS = gauss_seidel(A, x0, b, iterations=1)
    # r_norms_GS.append(np.linalg.norm(b - A @ x_GS))

    diff = x_noSmoother - x_pyamg
    print(f"max x_noSmoother diff:{np.max(diff)}, in {np.argmax(diff)}")
    # diff2 = x_rep - x_pyamg
    # print(f"max rep diff:{np.max(diff2)}, in {np.argmax(diff2)}")

    # ------------------------------- print results ------------------------------- #
    print_residuals(r_norms_pyamg, "pyamg")
    print_residuals(r_norms_rep, "rep")
    # print_residuals(r_norms_simplest, "simplest")
    # print_residuals(r_norms_repAnorm, "rep_Anorm")
    print_residuals(r_norms_noSmoother, "rep_noSmoother")
    # print_residuals(r_norms_direct, "direct")
    # print_residuals(r_norms_GS, "GS")
    # print_residuals(r_norms_FAS, "FAS")
    print_residuals(r_norms_AMG, "AMG")

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    plot_r_norms(r_norms_pyamg, axs[0], title=plot_title,linestyle="-",label="pyamg")
    # plot_r_norms(r_norms_direct, axs[0], linestyle="-.",label="direct")
    plot_r_norms(r_norms_rep, axs[0], title=plot_title, linestyle="--",label="reprodced pyamg")
    plot_r_norms(r_norms_noSmoother, axs[0], title=plot_title, linestyle="--",label="no smoother")
    plot_r_norms(r_norms_AMG, axs[0], title=plot_title, linestyle="--",label="AMG")
    # plot_r_norms(r_norms_GS, axs[0], title=plot_title, linestyle=":",label="GS")
    # plot_r_norms(r_norms_simplest, axs[0], title=plot_title, linestyle="-.",label="simplest")
    # plot_r_norms(r_norms_repAnorm, axs[1], title=plot_title, linestyle="-.",label="repr_Anorm")
    # plot_r_norms(r_norms_FAS, axs[1], title=plot_title, linestyle="-.",label="FAS")

    fig.canvas.manager.set_window_title(plot_title)
    plt.tight_layout()
    if save_fig_instad_of_show:
        plt.savefig(f"result/test/residuals_{plot_title}.png")
    else:
        plt.show()


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
    
    # eigens_S = scipy.sparse.linalg.eigs(S)
    # spec_radius_S = max(abs(eigens_S[0]))
    # print("eigens S:", eigens_S[0])
    # print("spec_radius S:", spec_radius_S)
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
        print(f"{name}[{i}] = {r:.3e}")



def solve_pyamg(ml, b, r_norms=[]):
    x = ml.solve(b, tol=1e-3, residuals=r_norms, maxiter=1)
    return x

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
        

def solve_rep_noSmoother(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
    maxiter = 1

    A2 = R @ A @ P

    # out_dir = f"./result/test/"
    # print(f"writting A2 to {out_dir}")
    # scipy.io.mmwrite(out_dir + f"A2.mtx", A2)

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

    coarse_residuals = []
    while True:  # it <= maxiter and normr >= tol:
        # gauss_seidel(A, x, b, iterations=1)  # presmoother

        residual = b - A @ x

        coarse_b = R @ residual  # restriction

        coarse_x = np.zeros_like(coarse_b)

        coarse_normr = np.linalg.norm(coarse_b)
        coarse_residuals.append(coarse_normr)

        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)

        coarse_normr = np.linalg.norm(np.linalg.norm(coarse_b - A2 @ coarse_x))
        coarse_residuals.append(coarse_normr)


        x += P @ coarse_x  # coarse grid correction

        # gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x, coarse_residuals
        if it == maxiter:
            return x, coarse_residuals

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

def plot_r_norms(data, ax, *args, **kwargs):
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
