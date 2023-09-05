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

# from pyamg.relaxation import make_system
# from pyamg import amg_core

sys.path.append(os.getcwd())


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


def test_amg():
    # ------------------------------- prepare data ------------------------------- #
    # generate_A_b_pyamg(n=10)
    A = mmread("A.mtx")
    A = A.tocsr()
    b = np.loadtxt("b.txt")

    # generate R by pyamg
    ml = pyamg.classical.ruge_stuben_solver(A, max_levels=2)  # construct the multigrid hierarchy
    # print("ml bulit")
    # P = ml.levels[0].P
    # R = ml.levels[0].R
    R = mmread("R.mtx")
    P = mmread("P.mtx")
    print(f"R: {R.shape}, P: {P.shape}")

    # ------------------------------- test solvers ------------------------------- #

    use_AMG = True
    use_pyamg = True
    use_pyamgmy = True
    use_symGS = False
    use_GS = True

    # pyamg
    if use_pyamg:
        print("pyamg")
        r_norm_list_pyamg = []
        t = perf_counter()
        x0 = np.zeros_like(b)
        x_pyamg = solve_pyamg(ml, b, r_norm_list_pyamg)
        t_pyamg = perf_counter() - t
        t = perf_counter()

    # my pyamg reproduction
    if use_pyamgmy:
        print("pyamgmy")
        r_norm_list_pyamgmy = []
        t = perf_counter()
        x0 = np.zeros_like(b)
        # change_smoothers(ml, presmoother=("jacobi"), postsmoother=("jacobi"))
        x_pyamgmy = solve_pyamg_my(A, b, x0, R, P, r_norm_list_pyamgmy)
        t_pyamgmy = perf_counter() - t
        t = perf_counter()

    # my symmetic gauss seidel
    if use_symGS:
        print("symGS")
        r_norm_list_symGS = []
        t = perf_counter()
        x0 = np.zeros_like(b)
        x_symGS = solve_gauss_seidel_symmetric_new(A, b, x0, 5, r_norm_list_symGS)
        t_symGS = perf_counter() - t
        t = perf_counter()

    if use_GS:
        print("GS")
        r_norm_list_GS = []
        t = perf_counter()
        x0 = np.zeros_like(b)
        x_GS = np.zeros_like(b)
        for _ in range(5):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x_GS, b, row_start=0, row_stop=int(len(x0)), row_step=1)
            r_norm = np.linalg.norm(A @ x_GS - b)
            r_norm_list_GS.append(r_norm)
        t_GS = perf_counter() - t
        t = perf_counter()

    # ------------------------------- plot ------------------------------- #
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    if use_pyamg:
        print(f"r pyamg: {r_norm_list_pyamg[0]:.2e}, {r_norm_list_pyamg[1]:.2e}")
    if use_pyamgmy:
        print(f"r pyamgmy:")
        print(*r_norm_list_pyamgmy)
    if use_symGS:
        print(f"r symGS: {r_norm_list_symGS[0]:.2e}, {r_norm_list_symGS[1]:.2e}")
    if use_GS:
        print(f"r GS:")
        print(*r_norm_list_GS)

    if use_pyamg:
        plot_r_norm_list(r_norm_list_pyamg, axs[0], "pyamg")
    if use_pyamgmy:
        plot_r_norm_list(r_norm_list_pyamgmy, axs[1], "pyamgmy")
    # if use_symGS:
    #     plot_r_norm_list(r_norm_list_symGS, axs[2], "symGS")
    if use_GS:
        plot_r_norm_list(r_norm_list_GS, axs[2], "GS")
    plt.tight_layout()
    plt.show()


def solve_pyamg(ml, b, r_norm_list=[]):
    x = ml.solve(b, tol=1e-3, residuals=r_norm_list, maxiter=1)
    return x


def solve_pyamg_my2(A, b, x0, R, P, r_norm_list=[]):
    tol = 1e-3
    residuals = r_norm_list
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


def solve_pyamg_my(A, b, x0, R, P, r_norm_list=[]):
    max_levels = 2

    tol = 1e-3
    residuals = r_norm_list
    maxiter = 1

    levels = []

    Level = namedtuple("Level", ["A", "R", "P", "presmoother", "postsmoother"])
    levels.append(Level(A, R, P, None, None))
    A2 = R @ A @ P
    levels.append(Level(A2, None, None, None, None))

    x = np.zeros_like(b)

    # Scale tol by normb
    # Don't scale tol earlier. The accel routine should also scale tol
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance

    # Start cycling (no acceleration)
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0

    while True:  # it <= maxiter and normr >= tol:
        if len(levels) == 1:
            # hierarchy has only 1 level
            # x = ml.coarse_solver(A, b)
            x = scipy.sparse.linalg.spsolve(A, b)
        else:
            __solve(levels, 0, x, b)

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)

        if normr < tol * normb:
            return x

        if it == maxiter:
            return x


def __solve(levels, lvl, x, b):
    A = levels[lvl].A

    # levels[lvl].presmoother(A, x, b)
    # x, _ = solve_gauss_seidel_symmetric(A, b, x, max_iterations=1)
    gauss_seidel(A, x, b, iterations=1)
    
    # np.savetxt("r1_after_presmooth.txt", b - A @ x)

    residual = b - A @ x

    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    # if lvl == len(levels) - 2:
    #     # coarse_x[:] = coarse_solver(levels[-1].A, coarse_b)
    #     coarse_x[:] = scipy.sparse.linalg.spsolve(levels[-1].A, coarse_b)
    # else:
    #     ...

    x += levels[lvl].P @ coarse_x  # coarse grid correction

    # np.savetxt("r1_after_prolongate.txt", b - A @ x)

    # levels[lvl].postsmoother(A, x, b)
    # x, _ = solve_gauss_seidel_symmetric(A, b, x, max_iterations=1)
    gauss_seidel(A, x, b, iterations=1)

    # np.savetxt("r1_after_postsmooth.txt", b - A @ x)


def gauss_seidel(A, x, b, iterations=1):
    if not sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        # forward sweep
        print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

        # backward sweep
        print("backward sweeping")
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


def solve_gauss_seidel_sparse(A, b, x0, max_iterations=100, tolerance=1e-6, r_norm_list=[]):
    # gauss seidel is just omega = 1 in SOR
    n = A.shape[0]
    x = x0.copy()

    for iter in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            Ax_new = A[i, :i].dot(x_new[:i])
            Ax_old = A[i, i + 1 :].dot(x[i + 1 :])
            x_new[i] = (1.0 / A[i, i]) * (b[i] - Ax_new - Ax_old)

        x = x_new.copy()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        r_norm_list.append(r_norm)
        print(f"GS iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def solve_gauss_seidel_symmetric(A, b, x0, max_iterations=1, tolerance=1e-6, r_norm_list=[]):
    # gauss seidel is just omega = 1 in SOR
    n = A.shape[0]
    x = x0.copy()
    print("Symmetric Gauss Seidel")

    print("forward sweeping")
    x, r = solve_gauss_seidel_sparse(A, b, x0, max_iterations=max_iterations, tolerance=1e-6, r_norm_list=r_norm_list)
    print("backward sweeping")
    x, r = solve_gauss_seidel_backward(A, b, x, max_iterations=max_iterations, tolerance=1e-6, r_norm_list=r_norm_list)

    return x, r


def solve_gauss_seidel_symmetric_new(A, x, b, iterations=1, r_norm_list=[]):
    if not sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        print(f"symGS iter {_iter}")
        # forward sweep
        print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)
            r_norm = np.linalg.norm(A @ x - b)
            r_norm_list.append(r_norm)

        # backward sweep
        print("backward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(
                A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
            )
            r_norm = np.linalg.norm(A @ x - b)
            r_norm_list.append(r_norm)
    return x


def solve_gauss_seidel_backward(A, b, x0, max_iterations=1, tolerance=1e-6, r_norm_list=[]):
    # gauss seidel is just omega = 1 in SOR
    n = A.shape[0]
    x = x0.copy()

    for iter in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            # Ax_new = A[i, :i].dot(x_new[:i])
            # Ax_old = A[i, i + 1 :].dot(x[i + 1 :])
            Ax_new = A[i, i + 1 :].dot(x_new[i + 1 :])
            Ax_old = A[i, :i].dot(x[:i])
            x_new[i] = (1.0 / A[i, i]) * (b[i] - Ax_new - Ax_old)

        x = x_new.copy()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        r_norm_list.append(r_norm)
        print(f"GS iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def coarse_solver(A2, r2):
    global ml
    return ml.coarse_solver(A2, r2)


def plot_r_norm_list(data, ax, title):
    x = np.arange(len(data))
    ax.plot(x, data, "-o")
    ax.set_title(title)
    # ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")


if __name__ == "__main__":
    test_amg()
