import taichi as ti
import numpy as np
import scipy
import scipy.sparse as sp
import os, sys
from time import perf_counter
from matplotlib import pyplot as plt
import matplotlib

sys.path.append(os.getcwd())

global r_norm_list
r_norm_list = []


def test_amg_vs_jacobian():
    # ------------------------------- data prepare ------------------------------- #
    print("loading data...")
    A = scipy.io.mmread("A.mtx")
    A = A.tocsr()
    b = np.loadtxt("b.txt")
    R = scipy.io.mmread("R.mtx")
    # P = scipy.io.mmread("P.mtx")
    # R = sp.random(1000, 34295, density=0.001, dtype=bool)
    P = R.transpose()
    print(f"R: {R.shape}, P: {P.shape}, A: {A.shape}, b1: {b.shape}")

    # ------------------------------- test solvers ------------------------------- #
    global r_norm_list

    # Direct solver
    r_norm_list.clear()
    t = perf_counter()
    x0 = np.zeros_like(b)
    x_direct = scipy.sparse.linalg.spsolve(A, b)
    t_direct = perf_counter() - t
    t = perf_counter()

    # AMG
    r_norm_list.clear()
    t = perf_counter()
    x0 = np.zeros_like(b)
    x_amg, r_amg = solve_amg(A, b, x0, R, P)
    t_amg = perf_counter() - t
    t = perf_counter()
    r_norm_list_amg = r_norm_list.copy()

    # Jacobi
    r_norm_list.clear()
    t = perf_counter()
    x0 = np.zeros_like(b)
    x_jacobi, r_jacobi = solve_jacobian_sparse(A, b, x0, 100, 1e-5)
    t_jacobi = perf_counter() - t
    t = perf_counter()
    r_norm_list_jacobi = r_norm_list.copy()

    # Gauss-Seidel
    r_norm_list.clear()
    t = perf_counter()
    x0 = np.zeros_like(b)
    x_gs, r_gs = solve_gauss_seidel_sparse(A, b, x0, 100, 1e-5)
    t_gs = perf_counter() - t
    t = perf_counter()
    r_norm_list_gs = r_norm_list.copy()

    # SOR
    r_norm_list.clear()
    t = perf_counter()
    x0 = np.zeros_like(b)
    x_sor, r_sor = solve_sor_sparse(A, b, x0, 1.25, 100, 1e-5)
    t_sor = perf_counter() - t
    t = perf_counter()
    r_norm_list_sor = r_norm_list.copy()

    # ------------------------------- print results ------------------------------- #
    print(f"Direct solver time: {t_direct:.2e}")
    print(f"AMG time: {t_amg:.2e}")
    print(f"Jacobian time: {t_jacobi:.2e}")
    print(f"Gauss-Seidel time: {t_gs:.2e}")
    print(f"SOR time: {t_sor:.2e}")

    print(f"Jacobi: max diff with direct solver: {np.linalg.norm(x_jacobi-x_direct, np.inf)}")
    assert np.allclose(x_jacobi, x_direct, atol=1e-4), f"max diff: {np.linalg.norm(x_jacobi-x_direct, np.inf)}"
    print(f"AMG: max diff with direct solver: {np.linalg.norm(x_amg-x_direct, np.inf)}")
    assert np.allclose(x_amg, x_direct, atol=1e-4), f"max diff: {np.linalg.norm(x_amg-x_direct, np.inf)}"
    print(f"Gauss-Seidel: max diff with direct solver: {np.linalg.norm(x_gs-x_direct, np.inf)}")
    assert np.allclose(x_gs, x_direct, atol=1e-4), f"max diff: {np.linalg.norm(x_gs-x_direct, np.inf)}"
    print(f"SOR: max diff with direct solver: {np.linalg.norm(x_sor-x_direct, np.inf)}")
    assert np.allclose(x_sor, x_direct, atol=1e-4), f"max diff: {np.linalg.norm(x_sor-x_direct, np.inf)}"
    print("All solutions is correct!\n")

    # ------------------------------- plot ------------------------------- #
    fig, axs = plt.subplots(2, 2, figsize=(10, 4))

    plot_r_norm_list(r_norm_list_amg, axs[0, 0], "AMG")
    plot_r_norm_list(r_norm_list_jacobi, axs[0, 1], "Jacobi")
    plot_r_norm_list(r_norm_list_gs, axs[1, 0], "Gauss-Seidel")
    plot_r_norm_list(r_norm_list_sor, axs[1, 1], "SOR Omega=1.25")

    plt.tight_layout()
    plt.show()


def jacobi_iteration(A, b, x0, max_iterations=100, tolerance=1e-6):
    global r_norm_list
    n = len(b)
    x = x0.copy()  # 初始解向量
    x_new = np.zeros_like(x)  # 存储更新后的解向量
    for iteration in range(max_iterations):
        for i in range(n):
            sum_term = np.dot(A[i, :n], x[:n]) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_term) / A[i, i]
        residual = b - np.dot(A, x_new)
        r_norm = np.linalg.norm(residual)

        r_norm_list.append(r_norm)

        if r_norm < tolerance:
            break
        print(f"iter: {iteration}, residual: {residual}")
        x = x_new.copy()
    return x_new, residual


def solve_amg(A1, b, x0, R, P):
    print("\n----------start AMG-----------")
    global r_norm_list

    # x1 initial guess
    x1 = x0
    r1 = b - A1 @ x1
    print(f"r1 initial:{np.linalg.norm(r1)}")
    r_norm_list.append(np.linalg.norm(r1))

    # 1. pre-smooth jacobian
    print(">>> 1. pre-smooth")
    print(f"r1 before pre-smooth:{np.linalg.norm(r1):.2e}")
    x1, r1 = solve_jacobian_sparse(A1, b, x1, max_iterations=50, tolerance=1e-2)
    print(f"r1 after pre-smooth:{np.linalg.norm(r1):.2e}")
    r_norm_list.append(np.linalg.norm(r1))

    # 2 restriction: pass r1 to r2 and construct A2
    print(">>> 2. restriction")
    # print(R.shape, r1.shape, P.shape, A1.shape)
    r2 = R @ r1
    A2 = R @ A1 @ P

    # 3 solve coarse level A2E2=r2
    print(">>> 3. solve coarse")
    E2 = scipy.sparse.linalg.spsolve(A2, r2)
    # E2 = np.linalg.solve(A2, r2)

    # 4 prolongation: get E1 and add to x1
    print(">>> 4. prolongate")
    E1 = P @ E2
    x1 += E1

    print(f"r1 before solve coarse:{ np.linalg.norm(r1):.2e}")
    r1 = b - A1 @ x1
    print(f"r1 after solve coarse:{ np.linalg.norm(r1):.2e}")
    r_norm_list.append(np.linalg.norm(r1))

    # 5 post-smooth jacobian
    print(">>> 5. post-smooth")
    print(f"r1 before post-smooth:{np.linalg.norm(r1):.2e}")
    x1, r1 = solve_jacobian_sparse(A1, b, x1, max_iterations=100, tolerance=1e-5)
    print(f"r1 after post-smooth:{np.linalg.norm(r1):.2e}")
    r_norm_list.append(np.linalg.norm(r1))

    x = x1
    print("----------finish AMG-----------\n")
    return x, r1


# ---------------------------------------------------------------------------- #
#                                 Ax=b solvers                                 #
# ---------------------------------------------------------------------------- #


def solve_jacobian_ti(A, b, x0, max_iterations=100, tolerance=1e-6):
    print("Solving Ax=b using Jacobian, taichi implementation...")
    n = A.shape[0]
    x = x0.copy()

    r = b - (A @ x)
    r_norm = np.linalg.norm(r)
    print(f"initial residual: {r_norm:.2e}")

    for iter in range(max_iterations):
        x_new = x.copy()
        jacobian_iter_once_kernel(A, b, x, x_new)
        x = x_new.copy()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        print(f"iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


@ti.kernel
def jacobian_iter_once_kernel(
    A: ti.types.ndarray(), b: ti.types.ndarray(), x: ti.types.ndarray(), x_new: ti.types.ndarray()
):
    n = b.shape[0]
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]
        x_new[i] = r / A[i, i]


def solve_jacobian_sparse(A, b, x0, max_iterations=100, tolerance=1e-6):
    global r_norm_list
    n = len(b)
    x = x0.copy()  # 初始解向量
    x_new = x0.copy()  # 存储更新后的解向量
    L = scipy.sparse.tril(A, k=-1)
    U = scipy.sparse.triu(A, k=1)
    D = A.diagonal()
    D_inv = 1.0 / D[:]
    D_inv = scipy.sparse.diags(D_inv)

    r = b - (A @ x_new)
    r_norm = np.linalg.norm(r)
    print(f"initial residual: {r_norm:.2e}")

    for iter in range(max_iterations):
        x_new = D_inv @ (b - (L + U) @ x)

        x = x_new.copy()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        print(f"iter {iter}, r={r_norm:.2e}")

        r_norm_list.append(r_norm)

        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


# # 定义稀疏矩阵的雅可比迭代函数
# def jacobi_iteration_sparse(A, b, x0, max_iterations=100, tolerance=1e-6):
#     n = len(b)
#     x = x0.copy()  # 初始解向量
#     x_new = np.zeros_like(x)  # 存储更新后的解向量
#     L = scipy.sparse.tril(A, k=-1)
#     U = scipy.sparse.triu(A, k=1)
#     D = A.diagonal()
#     D_inv = 1.0 / D[:]
#     D_inv = scipy.sparse.diags(D_inv)
#     for iteration in range(max_iterations):
#         x_new = D_inv @ (b - (L + U) @ x)

#         residual = b - (A @ x_new)
#         r_norm = np.linalg.norm(residual)
#         if r_norm < tolerance:
#             break
#         print(f"jacobian iter: {iteration}, residual: {r_norm}")
#         x = x_new.copy()
#     return x_new, residual


def solve_sor_sparse(A, b, x0, omega=1.5, max_iterations=100, tolerance=1e-6):
    global r_norm_list
    n = A.shape[0]
    x = x0.copy()
    for iter in range(max_iterations):
        new_x = np.copy(x)
        for i in range(A.shape[0]):
            start_idx = A.indptr[i]
            end_idx = A.indptr[i + 1]
            row_sum = A.data[start_idx:end_idx] @ new_x[A.indices[start_idx:end_idx]]
            x[i] = new_x[i] + omega * (b[i] - row_sum) / A.data[start_idx:end_idx].sum()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        r_norm_list.append(r_norm)
        print(f"iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def solve_sor(A, b, x0, omega=1.5, max_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    x = x0.copy()

    # D = np.diag(A)
    # L = np.tril(A, k=-1)
    # U = np.triu(A, k=1)
    # Lw = np.linalg.inv(D + omega * L) @ (- omega * U + (1 - omega) * D )
    # spectral_radius_Lw = max(abs(np.linalg.eigvals(Lw)))
    # print(f"spectral radius of Lw: {spectral_radius_Lw:.2f}")

    for iter in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (
                b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1 :], x[i + 1 :])
            )

        x = x_new.copy()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        print(f"iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def solve_sor_sparse_new(A, b, x0, omega=1.5, max_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    x = x0.copy()

    for iter in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            Ax_new = A[i, :i].dot(x_new[:i])
            Ax_old = A[i, i + 1 :].dot(x[i + 1 :])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - Ax_new - Ax_old)

        x = x_new.copy()

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        print(f"iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def solve_direct_solver(A, b):
    t = perf_counter()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)
    print(f"time: {perf_counter() - t}")
    print(f"shape of A: {A.shape}")
    print(f"solve success: {solver.info()}")
    return x


@ti.kernel
def gauss_seidel_kernel(A: ti.types.ndarray(), b: ti.types.ndarray(), x: ti.types.ndarray(), xOld: ti.types.ndarray()):
    N = b.shape[0]
    for i in range(N):
        entry = b[i]
        diagonal = A[i, i]
        if ti.abs(diagonal) < 1e-10:
            print("Diagonal element is too small")

        for j in range(i):
            entry -= A[i, j] * x[j]
        for j in range(i + 1, N):
            entry -= A[i, j] * xOld[j]
        x[i] = entry / diagonal


def solve_gauss_seidel_ti(A, b, x0, max_iterations=100, tolerance=1e-6):
    x = x0.copy()
    for iter in range(max_iterations):
        xOld = x.copy()
        gauss_seidel_kernel(A, b, x, xOld)

        # 计算残差并检查收敛
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        print(f"iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def solve_gauss_seidel_sparse(A, b, x0, max_iterations=100, tolerance=1e-6):
    global r_norm_list
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
        print(f"iter {iter}, r={r_norm:.2e}")
        if r_norm < tolerance:
            print(f"Converged after {iter + 1} iterations. Final residual: {r_norm:.2e}")
            return x, r

    print("Did not converge within the maximum number of iterations.")
    print(f"Final residual: {r_norm:.2e}")
    return x, r


def plot_r_norm_list(data, ax, title):
    x = np.arange(len(data))
    ax.plot(x, data, "-o")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())


if __name__ == "__main__":
    test_amg_vs_jacobian()
