import taichi as ti
import numpy as np
import scipy
import scipy.sparse as sp
import os, sys
from time import time

sys.path.append(os.getcwd())


def test_amg_vs_jacobian():
    print("loading data...")
    A1 = scipy.io.mmread("A1.mtx")
    b1 = np.loadtxt("b1.txt")
    R = scipy.io.mmread("R.mtx")
    # P = scipy.io.mmread("P.mtx")
    # R = sp.random(1000, 34295, density=0.001, dtype=bool)
    P = R.transpose()
    print(f"R: {R.shape}, P: {P.shape}, A1: {A1.shape}, b1: {b1.shape}")
    t = time()
    amg_step(A1, b1, R, P)
    print(f"AMG time: {time() - t}")


# def jacobi_iteration(A, b, x0, max_iterations=100, tolerance=1e-6):
#     n = len(b)
#     x = x0.copy()  # 初始解向量
#     x_new = np.zeros_like(x)  # 存储更新后的解向量
#     for iteration in range(max_iterations):
#         for i in range(n):
#             sum_term = np.dot(A[i, :n], x[:n]) - A[i, i] * x[i]
#             x_new[i] = (b[i] - sum_term) / A[i, i]
#         residual = np.linalg.norm(b - np.dot(A, x_new))
#         if np.linalg.norm(residual) < tolerance:
#             break
#         print(f"iter: {iteration}, residual: {residual}")
#         x = x_new.copy()
#     return x_new, residual


# 定义稀疏矩阵的雅可比迭代函数
def jacobi_iteration_sparse(A, b, x0, max_iterations=100, tolerance=1e-6):
    n = len(b)
    x = x0.copy()  # 初始解向量
    x_new = np.zeros_like(x)  # 存储更新后的解向量
    L = scipy.sparse.tril(A, k=-1)
    U = scipy.sparse.triu(A, k=1)
    D = A.diagonal()
    D_inv = 1.0 / D[:]
    D_inv = scipy.sparse.diags(D_inv)
    for iteration in range(max_iterations):
        x_new = D_inv @ (b - (L + U) @ x)

        residual = b - (A @ x_new)
        r_norm = np.linalg.norm(residual)
        if r_norm < tolerance:
            break
        print(f"jacobian iter: {iteration}, residual: {r_norm}")
        x = x_new.copy()
    return x_new, residual


def amg_step(A1, b1, R, P):
    # pre-smooth jacobian:
    print(f"before jacobian pre-smooth: ")
    print(f"b1:{np.linalg.norm(b1)}")
    x1, r1 = jacobi_iteration_sparse(A1, b1, b1, 3)

    # restriction: pass r1 to r2 and construct A2
    r2 = R @ r1
    A2 = R @ A1 @ P

    # solve coarse level A2E2=r2
    E2 = scipy.sparse.linalg.spsolve(A2, r2)

    # prolongation:
    E1 = P @ E2
    x1 += E1

    r1_new = b1 - A1 @ x1
    # post-smooth jacobian:
    print(f"before jacobian post-smooth: ")
    print(f"new r1:{np.linalg.norm(r1_new)}")
    x1, r1_new = jacobi_iteration_sparse(A1, b1, x1, 3)

    r1_new = b1 - A1 @ x1
    print(f"b1:{np.linalg.norm(b1)}, r1: {np.linalg.norm(r1)}, r1_new: {np.linalg.norm(r1_new)}")

    # with open("r1.txt", "a") as f:
    #     f.write(f"{np.linalg.norm(r1)}\n")
    # with open("r1_new.txt", "a") as f:
    #     f.write(f"{np.linalg.norm(r1_new)}\n")


if __name__ == "__main__":
    test_amg_vs_jacobian()
