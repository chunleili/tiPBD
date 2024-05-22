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
to_read_dir = prj_dir + "result/latest/A/"

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=100)
parser.add_argument("-title", type=str, default=f"")
parser.add_argument("-f", type=int, default=10)
N = parser.parse_args().N
print(f"N={N}")
plot_title = parser.parse_args().title
frame = parser.parse_args().f
save_fig_instad_of_show = True
generate_data = False
show_plot = True


def test_amg(mat_size = 10, case_num = 0, postfix=""):
    # ------------------------------- prepare data ------------------------------- #
    if(generate_data):
        print("generating data...")
        # A, b = generate_A_b_pyamg(n=mat_size)
        A, b = generate_A_b_spd(n=mat_size)
        scipy.io.mmwrite(to_read_dir + f"A{case_num}.mtx", A)
        np.savetxt(to_read_dir + f"b{case_num}.txt", b)
    else:
        print("loading data...")
        A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
        A = A.tocsr()
        A = A.astype(np.float32)
        b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float32)


    # # codition_number_of_A = np.linalg.cond(A.toarray())
    # # print(f"condition number of A: {codition_number_of_A}")

    # eigenvalues_A_largest = scipy.sparse.linalg.eigsh(A,return_eigenvectors=False, which="LM", k=1)
    # print("largest eigenvalues of A:", eigenvalues_A_largest)
    # eigenvalues_A_smallest = scipy.sparse.linalg.eigsh(A,return_eigenvectors=False, which="SM", k=1)
    # print("smallest eigenvalues of A:", eigenvalues_A_smallest)
    # print("largest/smallest:", eigenvalues_A_largest/eigenvalues_A_smallest)

    # exit(0)

    ml = pyamg.ruge_stuben_solver(A)
    res = []
    x_pyamg = ml.solve(b, tol=1e-10, residuals=res,maxiter=300)
    print(ml)
    print("res1 classical AMG", len(res), res[-1])
    print((res[-1]/res[0])**(1.0/(len(res)-1)))


    ml2 = pyamg.smoothed_aggregation_solver(A)
    res2 = []
    x_pyamg2 = ml2.solve(b, tol=1e-10, residuals=res2,maxiter=300)
    # print(ml2)
    print("res2 SA", len(res2), res2[-1])
    print((res2[-1]/res2[0])**(1.0/(len(res2)-1)))

    # GS
    x4 = np.zeros_like(b)
    res4 = []
    for _ in range(300):
        res4.append(np.linalg.norm(b - A @ x4))
        x4 = gauss_seidel(A, x4, b, iterations=1)
        x4 = gauss_seidel(A, x4, b, iterations=1)
    print("res4 GS",len(res4), res4[-1])
    print((res4[-1]/res4[0])**(1.0/(len(res4)-1)))

    # from diagnostic, SA+CG
    ml5 = pyamg.smoothed_aggregation_solver(A, B=b, BH=b.copy(),
        strength=('symmetric', {'theta': 0.0}),
        smooth="jacobi",
        improve_candidates=None,
        aggregate="standard",
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        max_levels=15,
        max_coarse=300,
        coarse_solver="pinv")
    x0 = np.zeros_like(b)
    res5 = []
    x5 = ml5.solve(b, x0=x0, tol=1e-08, residuals=res5, accel="cg", maxiter=300, cycle="W")
    print("res5 SA+CG",len(res5),res5[-1])
    print((res5[-1]/res5[0])**(1.0/(len(res5)-1)))

    # CG
    x6 = np.zeros_like(b)
    res6 = []
    res6.append(np.linalg.norm(b - A @ x6))
    x6 = scipy.sparse.linalg.cg(A, b, x0=x6, tol=1e-10, maxiter=300, callback=lambda x: res6.append(np.linalg.norm(b - A @ x)))
    print("CG res6",len(res6),res6[-1])
    print((res6[-1]/res6[0])**(1.0/(len(res6)-1)))

    #  diagnal preconditioner + CG
    M = scipy.sparse.diags(1.0/A.diagonal())
    x7 = np.zeros_like(b)
    res7 = []
    res7.append(np.linalg.norm(b - A @ x7))
    x7 = scipy.sparse.linalg.cg(A, b, x0=x7, tol=1e-10, maxiter=300, callback=lambda x: res7.append(np.linalg.norm(b - A @ x)), M=M)
    print("res7 diag+CG", len(res7),res7[-1])
    print((res7[-1]/res7[0])**(1.0/(len(res7)-1)))


    if show_plot:
        fig, axs = plt.subplots(1, figsize=(8, 9))
        plot_residuals(res/res[0], axs,  label="Classical AMG", marker="o", color="blue")
        plot_residuals(res2/res2[0], axs,  label="Smoothed Aggregation", marker="x", color="orange")
        plot_residuals(res4/res4[0], axs,  label="Gauss Seidel", marker="s", color="red")
        plot_residuals(res5/res5[0], axs,  label="SA+CG", marker="d", color="purple")
        plot_residuals(res6/res6[0], axs,  label="CG", marker="^", color="green")
        plot_residuals(res7/res7[0], axs,  label="diag CG", marker="v", color="black")
        plot_title = postfix
        fig.canvas.manager.set_window_title(plot_title)
        plt.tight_layout()
        if save_fig_instad_of_show:
            plt.savefig(f"result/latest/residuals_{plot_title}.png")
        else:
            plt.show()


def test_amg1(mat_size = 10, case_num = 0, postfix=""):
    # ------------------------------- prepare data ------------------------------- #
    if(generate_data):
        print("generating data...")
        # A, b = generate_A_b_pyamg(n=mat_size)
        A, b = generate_A_b_spd(n=mat_size)
        scipy.io.mmwrite(to_read_dir + f"A{case_num}.mtx", A)
        np.savetxt(to_read_dir + f"b{case_num}.txt", b)
    else:
        print("loading data...")
        A = scipy.io.mmread(to_read_dir+f"A_F10-0.mtx")
        A = A.tocsr()
        b = np.loadtxt(to_read_dir+f"b_F10-0.txt", dtype=np.float32)
        # b = np.random.random(A.shape[0])
        # b = np.ones(A.shape[0])

    A1 = A.copy()
    A2 = improve_A_by_remove_offdiag(A)
    A3 = improve_A_by_reduce_offdiag(A)
    t = perf_counter()
    print("to make M matrix...")
    A4 = improve_A_make_M_matrix(A)
    print(f"make M matrix took {perf_counter() - t:.3e} s")
    print(f"A: {A.shape}")

    # generate R by pyamg
    R1,P1 = generate_R_P(A1)
    R2,P2 = generate_R_P(A2)
    R3,P3 = generate_R_P(A3)
    R4,P4 = generate_R_P(A4)
    scipy.io.mmwrite(to_read_dir + f"R{case_num}.mtx", R1)

    # analyse_A(A,R,P)

    # ------------------------------- test solvers ------------------------------- #
    # print("Solving pyamg...")
    # x0 = np.zeros_like(b)
    # res = []
    #ml = pyamg.ruge_stuben_solver(A, max_levels=2)
    #_,res = timer_wrapper(solve_pyamg, ml, b)

    x0 = np.zeros_like(b)
    x_amg = solve_amg(A, b, x0, R1, P1, residuals=[])
    x_rep,residuals_rep, full_residual_rep = timer_wrapper(solve_rep, A, b, x0, R1, P1)
    x_onlySmoother,residuals_onlySmoother = timer_wrapper(solve_onlySmoother, A, b, x0, R1, P1)
    x_noSmoother,residuals_noSmoother = timer_wrapper(solve_rep_noSmoother, A, b, x0, R1, P1)
    x_remove_offdiag,residuals_remove_offdiag,_ = timer_wrapper(solve_rep, A2, b, x0, R2, P2)
    x_reduce_offdiag,residuals_reduce_offdiag,_ = timer_wrapper(solve_rep, A3, b, x0, R3, P3)
    x_M_matrix,residuals_M_matrix,_ = timer_wrapper(solve_rep, A4, b, x0, R4, P4)

    #assert np.allclose(x_rep, x_amg, atol=1e-5)
    # print("generating R and P by selecting row...")
    # R2 = scipy.sparse.csr_matrix((2,A.shape[0]), dtype=np.int32)
    # R2[0,0] = 1
    # R2[1,9] = 1
    # P2 = R2.T
    # x0 = np.zeros_like(b)
    # _,residuals_selectRows = timer_wrapper(solve_rep_noSmoother, A, b, x0, R2, P2)

    # print("generating R and P by removing rows...")
    # R3 = scipy.sparse.identity(A.shape[0], dtype=np.int32)
    # R3=R3.tocsr()
    # R3 = delete_rows_csr(R3, range(0, A.shape[0] - 1, 2))
    # P3 = R3.T
    # print(f"##########R: {R3.shape}, P: {P3.shape}")
    # x0 = np.zeros_like(b)
    # print("rank of P3:", np.linalg.matrix_rank(P3.toarray()))
    # _,residuals_removeRows = timer_wrapper(solve_rep_noSmoother, A, b, x0, R3, P3)

    # ------------------------------- print results ---------------------------- #
    # print("x_rep:", x_rep)
    # x_rep_max = np.max(np.abs(x_rep))
    # print("x_onlySmoother:", np.max(np.abs(x_rep-x_onlySmoother)/x_rep_max))
    # print("x_noSmoother:", np.max(np.abs(x_rep-x_noSmoother)/x_rep_max))
    # print("x_remove_offdiag:", np.max(np.abs(x_rep-x_remove_offdiag)/x_rep_max))
    # print("x_reduce_offdiag:", np.max(np.abs(x_rep-x_reduce_offdiag)/x_rep_max))
    # print("x_M_matrix:", np.max(np.abs(x_rep-x_M_matrix)/x_rep_max))


    print_residuals(residuals_rep, "rep")
    print_residuals(residuals_onlySmoother, "onlySmoother")
    print_residuals(residuals_noSmoother, "noSmoother")
    print_residuals(residuals_remove_offdiag, "remove_offdiag")
    print_residuals(residuals_reduce_offdiag, "reduce_offdiag")
    print_residuals(residuals_M_matrix, "M_matrix")

    if show_plot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 9))
        plot_residuals(residuals_rep, axs[0], label="rep")
        plot_residuals(residuals_onlySmoother, axs[0], label="onlySmoother")
        plot_residuals(residuals_remove_offdiag, axs[0],  label="remove_offdiag")
        plot_residuals(residuals_reduce_offdiag, axs[0],  label="reduce_offdiag")
        plot_residuals(residuals_M_matrix, axs[0],  label="M_matrix")
        plot_residuals(residuals_noSmoother, axs[1],  label="noSmoother")

        # plot_full_residual(full_residual_rep[0], "residual0")
        # plot_full_residual(full_residual_rep[1], "residual1")
        # plot_full_residual(full_residual_rep[2], "residual2")
        # plot_full_residual(full_residual_rep[3], "residual3")

        fig.canvas.manager.set_window_title(plot_title)
        plt.tight_layout()
        if save_fig_instad_of_show:
            plt.savefig(f"result/latest/residuals_{plot_title}.png")
        else:
            plt.show()

def plot_full_residual(data, title=""):
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    N = np.sqrt(len(data)).astype(int)

    A = np.linspace(1, N, N)
    B = np.linspace(1, N, N)

    X, Y = np.meshgrid(A, B)
    d0 = data[:N*N].reshape((N, N))

    # Plot the surface.
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    surf0 = ax.plot_surface(X, Y, d0, cmap=cm.coolwarm, label="residual0")
    # ax.set_zlim(-.03, .03)
    fig.text(0.5, 0.9, title, ha='center')
    fig.canvas.manager.set_window_title(title)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf0, shrink=0.5, aspect=5)

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
    x = ml.solve(b, tol=1e-3, residuals=residuals, maxiter=1)
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


def solve_rep(A, b, x0, R, P):
    tol = 1e-3
    maxiter = 1
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



def solve_onlySmoother(A, b, x0, R, P):
    tol = 1e-3
    maxiter = 1
    residuals = []

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




def solve_amg(A, b, x0, R, P, residuals=[]):
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


def gauss_seidel(A, x, b, iterations=1):
    if not sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        # forward sweep
        # print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

        # backward sweep
        # print("backward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel_kernel(
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

def plot_residuals(data, ax, *args, **kwargs):
    title = kwargs.pop("title", "")
    linestyle = kwargs.pop("linestyle", "-")
    label = kwargs.pop("label", "")
    x = np.arange(len(data))
    ax.plot(x, data, label=label, linestyle=linestyle, *args, **kwargs)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.legend(loc="upper right")



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

if __name__ == "__main__":
    # test_all_A()
    for i in range(1,30,5):
        test_amg(10,0,f"F{i}-0")
    # test_amg(10,0,f"F1-0")
    # test_different_N()
