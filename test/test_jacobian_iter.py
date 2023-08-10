import taichi as ti
import random
import numpy as np
import scipy

ti.init(default_fp=ti.f64)


def jacobian_iter_solve(A, b, x, max_iterations=100, tolerance=1e-6):
    @ti.kernel
    def iter_once(A: ti.types.ndarray(), b: ti.types.ndarray(), x: ti.types.ndarray()):
        n = b.shape[0]
        for i in range(n):
            r = b[i]
            for j in range(n):
                if i != j:
                    r -= A[i, j] * x[j]

            x[i] = r / A[i, i]

    @ti.kernel
    def residual(A: ti.types.ndarray(), b: ti.types.ndarray(), x: ti.types.ndarray()) -> ti.f32:
        n = b.shape[0]
        res = 0.0

        for i in range(n):
            r = b[i] * 1.0
            for j in range(n):
                r -= A[i, j] * x[j]
            res += r * r

        return res

    def jacobian_iter(A, b, x, max_iterations=100, tolerance=1e-6):
        # print_A(A)
        res = residual(A, b, x)
        print(f"initial residual={res:0.10f}")
        i = 0
        while res > tolerance and i < max_iterations:
            i += 1
            iter_once(A, b, x)
            res = residual(A, b, x)
            print(f"iter {i}, residual={res:0.10f}")

    jacobian_iter(A, b, x, 100, 1e-6)
    return x


# ---------------------------------------------------------------------------- #
#                               data preparation                               #
# ---------------------------------------------------------------------------- #


@ti.kernel
def print_A(A: ti.types.ndarray()):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            print("A[", i, ",", j, "] = ", A[i, j])


def prepare_A_b_x():
    n = 20
    A = np.zeros(shape=(n, n), dtype=np.float64)
    x = np.zeros(shape=(n), dtype=np.float64)
    b = np.zeros(shape=(n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            A[i, j] = random.random() - 0.5

        A[i, i] += n * 0.1

        b[i] = random.random() * 100
    return A, b, x


# ---------------------------------------------------------------------------- #
#                             sparse matrix version                            #
# ---------------------------------------------------------------------------- #
def jacobi_iteration_sparse(A, b, x0, max_iterations=20, tolerance=1e-6, relative_tolerance=1e-12):
    n = len(b)
    x = x0.copy()  # 初始解向量
    x_new = x0.copy()  # 存储更新后的解向量
    L = scipy.sparse.tril(A, k=-1)
    U = scipy.sparse.triu(A, k=1)
    D = A.diagonal()
    D_inv = 1.0 / D[:]
    D_inv = scipy.sparse.diags(D_inv)

    residual = b - (A @ x_new)
    r_norm = np.linalg.norm(residual)
    print(f"initial residual: {r_norm}")

    for iteration in range(max_iterations):
        x_new = D_inv @ (b - (L + U) @ x)

        residual = b - (A @ x_new)
        r_norm = np.linalg.norm(residual)

        if r_norm < tolerance:
            print(f"reach abs tolerance at iter: {iteration}")
            break
        elif r_norm < relative_tolerance * np.linalg.norm(b):
            print(f"reach relative tolerance at iter: {iteration}")
            break
        elif r_norm > 1e10:
            print(f"diverge at iter: {iteration}")
            break
        print(f"jacobian iter: {iteration}, residual: {r_norm}")
        x = x_new.copy()

    if iteration == max_iterations - 1:
        print(f"reach max iterations")

    return x_new, residual


if __name__ == "__main__":
    A, b, x0 = prepare_A_b_x()

    # solving with dense matrix in taichi:
    x = x0.copy()
    x_ti = jacobian_iter_solve(A, b, x, 100, 1e-6)
    print("taichi solution: ", x_ti)

    # solving with sparse matrix in scipy:
    x = x0.copy()
    A_sp = scipy.sparse.csr_matrix(A)
    x_sp, r = jacobi_iteration_sparse(A_sp, b, x, 100, 1e-6)
    print("sparse solution: ", x_sp)

    # true solution:
    x = x0.copy()
    x_ture = np.linalg.solve(A, b)
    print("true solution: ", x_ture)

    assert np.allclose(x_ti, x_ture, rtol=1e-3)
    assert np.allclose(x_sp, x_ture, rtol=1e-3)
    print("test passed!")
