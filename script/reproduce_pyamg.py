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

parser = argparse.ArgumentParser()
parser.add_argument("-title", type=str, default=f"")
parser.add_argument("-f", type=int, default=20)
parser.add_argument("-case_name", type=str, default='scale64')
plot_title = parser.parse_args().title
frame = parser.parse_args().f
case_name = parser.parse_args().case_name

to_read_dir = prj_dir + f"result/{case_name}/A/"
save_fig = True
show_fig = True
generate_data = False
draw_plot = True
maxiter = 300
early_stop = False
tol=1e-10 # relative tolerance
run_concate_png = True
run_strength_options = False
postfix = ''

Residual = namedtuple('Residual', ['label','r', 't'])

def test_amg(A, b, postfix=""):
    # x0 = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    allres = []
    tic = perf_counter()

    # classical AMG
    label = "Classical AMG"
    print(f"Calculating {label}...")
    ml1 = pyamg.ruge_stuben_solver(A)
    r = []
    _ = ml1.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter)
    allres.append(Residual(label, r, perf_counter()))

    # SA
    label = "Smoothed Aggregation"
    print(f"Calculating {label}...")
    ml2 = pyamg.smoothed_aggregation_solver(A)
    r = []
    _ = ml2.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter)
    allres.append(Residual(label, r, perf_counter()))

    # Jacobi: diverge
    # x3 = x0.copy()
    # res3 = []
    # for _ in range(maxiter+1):
    #     res3.append(np.linalg.norm(b - A @ x3))
    #     pyamg.relaxation.relaxation.jacobi(A=A, x=x3, b=b, iterations=1)
    # conv3 = calc_conv(res3)
    # print("res3 Jacobi",conv3)
    # toc3 = perf_counter()

    # GS
    label = "Gauss Seidel"
    print(f"Calculating {label}...")
    x4 = x0.copy()
    r = []
    for _ in range(maxiter*8+1):
        r.append(np.linalg.norm(b - A @ x4))
        pyamg.relaxation.relaxation.gauss_seidel(A=A, x=x4, b=b, iterations=1)
    allres.append(Residual(label, r, perf_counter()))

    #  SA+CG, from diagnostic,
    label = "SA+CG"
    print(f"Calculating {label}...")
    r = []
    B = np.ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()
    ml5 = pyamg.smoothed_aggregation_solver(A,B=B,BH=BH, 
        strength=('symmetric', {'theta': 0.0}),
        smooth="jacobi",
        improve_candidates=None,
        aggregate="standard",
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        max_levels=15,
        max_coarse=300,
        coarse_solver="pinv")
    x = ml5.solve(b, x0=x0, tol=tol, residuals=r, accel="cg", maxiter=maxiter, cycle="W")
    allres.append(Residual(label, r, perf_counter()))

    # CG
    label = "CG"
    print(f"Calculating {label}...")
    x6 = x0.copy()
    r = []
    r.append(np.linalg.norm(b - A @ x6))
    x6 = scipy.sparse.linalg.cg(A, b, x0=x0.copy(), rtol=tol, maxiter=maxiter, callback=lambda x: r.append(np.linalg.norm(b - A @ x)))
    allres.append(Residual(label, r, perf_counter()))

    #  diagnal preconditioner + CG
    label = "diag PCG"
    print(f"Calculating {label}...")
    M = scipy.sparse.diags(1.0/A.diagonal())
    x7 = x0.copy()
    r = []
    r.append(np.linalg.norm(b - A @ x7))
    x7 = scipy.sparse.linalg.cg(A, b, x0=x0.copy(), rtol=tol, maxiter=maxiter, callback=lambda x: r.append(np.linalg.norm(b - A @ x)), M=M)
    allres.append(Residual(label, r, perf_counter()))

    # SA with strength algebraic_distance_epsilon3
    label = "SA+CG+Algebraic3.0"
    print(f"Calculating {label}...")
    ml8 = pyamg.smoothed_aggregation_solver(A, max_coarse=300, max_levels=15, strength=('algebraic_distance', {'epsilon': 3.0}))
    r = []
    _ = ml8.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))


    # SA with strength affinity_4.0
    label = "SA+CG+Affinity4.0"
    print(f"Calculating {label}...")
    ml9 = pyamg.smoothed_aggregation_solver(A, max_coarse=300, max_levels=15, strength=('affinity', {'epsilon': 4.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    r = []
    _ = ml9.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))


    # blackbox
    label = "Blackbox"
    print(f"Calculating {label}...")
    r=[]
    x = pyamg.solve(A, b, x0, tol=tol, verb=False, residuals=r, maxiter=maxiter)
    conv10 = calc_conv(r)
    allres.append(Residual(label, r, perf_counter()))

    # rootnode
    label = "Rootnode+CG"
    print(f"Calculating {label}...")
    ml12 = pyamg.rootnode_solver(A)
    r = []
    x12 = ml12.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))


    # SA+CG normal
    label = "SA+CG normal"
    print(f"Calculating {label}...")
    ml13 = pyamg.smoothed_aggregation_solver(A)
    r = []
    _ = ml13.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))

    # SA+CG smooth='energy'
    label = "SA+CG smooth=energy"
    print(f"Calculating {label}...")
    ml14 = pyamg.smoothed_aggregation_solver(A, smooth='energy')
    r = []
    _ = ml14.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))

    # CAMG+CG
    label = "CAMG+CG"
    print(f"Calculating {label}...")
    ml16 = pyamg.ruge_stuben_solver(A)
    r = []
    _ = ml16.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))

    # UA+CG
    label = "UA+CG"
    print(f"Calculating {label}...")
    ml17 = pyamg.smoothed_aggregation_solver(A, smooth=None)
    r = []
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))

    label = "UA+CG coarse=GS"
    print(f"Calculating {label}...")
    ml18 = pyamg.smoothed_aggregation_solver(A, smooth=None, coarse_solver='gauss_seidel', max_coarse=300)
    r = []
    _ = ml18.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))

    convs,times,labels  = postprocess_residual(allres, tic)
    
    draw_convergence_factors(convs, labels)
    draw_times(times, labels)

    df = print_df(labels, convs, times)
    save_data(allres,postfix)

    if draw_plot:
        colors = ['blue', 'orange', 'red', 'purple', 'green', 'black', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'brown', 'pink']
        markers = ['o', 'x', 's', 'd', '^', 'v', '>', '<', '1', '2', '3', '4', '+', 'X']
        markers = ['' for _ in range(len(allres)-1)]
        markers.append('o')

        # https://matplotlib.org/stable/api/markers_api.html for different markers
        # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def for different colors
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        fig, axs = plt.subplots(1, figsize=(8, 9))
        for i in range(len(allres)):
            # if allres[i].label == 'SA+CG' or\
            #    allres[i].label == 'UA+CG' or\
            #    allres[i].label == 'UA+CG coarse=GS':
            plot_residuals(a2r(allres[i].r), axs,  label=allres[i].label, marker=markers[i], color=colors[i])

        global plot_title
        plot_title = postfix
        fig.canvas.manager.set_window_title(plot_title)
        plt.tight_layout()
        if save_fig:
            dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
            mkdir_if_not_exist(dir)
            plt.savefig(dir+f"/residuals_{plot_title}.png")
        if show_fig:
            plt.show()

def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))

def a2r(r): #absolute to relative
    return r/r[0]

def draw_convergence_factors(convs, labels):
    assert len(convs) == len(labels)
    print("\n\nConvergence factor of each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{convs[i]:.3f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(convs)), convs, color='blue')
    ax.set_yticks(range(len(convs)))
    ax.set_yticklabels(labels)
    ax.set_title("Convergence factor of each solver")



def draw_times(times, labels):
    assert len(times) == len(labels)
    print("\n\nTime(s) taken for each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{times[i]:.2f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(times)), times, color='red')
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(labels)
    ax.set_title("Time taken for each solver")


def print_df(labels, convs, times, verbose=False):
    import pandas as pd
    print("\n\nDataframe of convergence factor and time taken for each solver")
    pd.set_option("display.precision", 3)
    df = pd.DataFrame({"label":labels, "conv_fac":convs, "time":times})
    print(df)
    if verbose:
        print("\nIn increasing order of conv_fac:")
        df = df.sort_values(by="conv_fac", ascending=True)
        print(df)
        print("\nIn increasing order of time taken:")
        df = df.sort_values(by="time", ascending=True)
        print(df)
    return df

def save_data(allres, postfix=""):
    import pandas as pd
    df = pd.DataFrame(allres)
    dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
    mkdir_if_not_exist(dir)
    df.to_csv(dir+f"/allres_{postfix}.csv")

def postprocess_residual(allres, tic):
    # import pandas as pd
    #calculate convergence factor and time
    convs = np.zeros(len(allres))
    times = np.zeros(len(allres)+1)
    times[0] = tic
    for i in range(len(allres)):
        convs[i] = calc_conv(allres[i].r)
        times[i+1] = allres[i].t
    times = np.diff(times)
    for i in range(len(allres)):
        allres[i]._replace(t = times[i])
    labels = [ri.label for ri in allres]
    return convs, times, labels


# def test_amg1(mat_size = 10, case_num = 0, postfix=""):
#     # ------------------------------- prepare data ------------------------------- #
#     if(generate_data):
#         print("generating data...")
#         # A, b = generate_A_b_pyamg(n=mat_size)
#         A, b = generate_A_b_spd(n=mat_size)
#         scipy.io.mmwrite(to_read_dir + f"A{case_num}.mtx", A)
#         np.savetxt(to_read_dir + f"b{case_num}.txt", b)
#     else:
#         print("loading data...")
#         A = scipy.io.mmread(to_read_dir+f"A_F10-0.mtx")
#         A = A.tocsr()
#         b = np.loadtxt(to_read_dir+f"b_F10-0.txt", dtype=np.float32)
#         # b = np.random.random(A.shape[0])
#         # b = np.ones(A.shape[0])

#     A1 = A.copy()
#     A2 = improve_A_by_remove_offdiag(A)
#     A3 = improve_A_by_reduce_offdiag(A)
#     t = perf_counter()
#     print("to make M matrix...")
#     A4 = improve_A_make_M_matrix(A)
#     print(f"make M matrix took {perf_counter() - t:.3e} s")
#     print(f"A: {A.shape}")

#     # generate R by pyamg
#     R1,P1 = generate_R_P(A1)
#     R2,P2 = generate_R_P(A2)
#     R3,P3 = generate_R_P(A3)
#     R4,P4 = generate_R_P(A4)
#     scipy.io.mmwrite(to_read_dir + f"R{case_num}.mtx", R1)

#     # analyse_A(A,R,P)

#     # ------------------------------- test solvers ------------------------------- #
#     # print("Solving pyamg...")
#     # x0 = np.zeros_like(b)
#     # res = []
#     #ml = pyamg.ruge_stuben_solver(A, max_levels=2)
#     #_,res = timer_wrapper(solve_pyamg, ml, b)

#     x0 = np.zeros_like(b)
#     x_amg = solve_amg(A, b, x0, R1, P1, residuals=[])
#     x_rep,residuals_rep, full_residual_rep = timer_wrapper(solve_rep, A, b, x0, R1, P1)
#     x_onlySmoother,residuals_onlySmoother = timer_wrapper(solve_onlySmoother, A, b, x0, R1, P1)
#     x_noSmoother,residuals_noSmoother = timer_wrapper(solve_rep_noSmoother, A, b, x0, R1, P1)
#     x_remove_offdiag,residuals_remove_offdiag,_ = timer_wrapper(solve_rep, A2, b, x0, R2, P2)
#     x_reduce_offdiag,residuals_reduce_offdiag,_ = timer_wrapper(solve_rep, A3, b, x0, R3, P3)
#     x_M_matrix,residuals_M_matrix,_ = timer_wrapper(solve_rep, A4, b, x0, R4, P4)

#     #assert np.allclose(x_rep, x_amg, atol=1e-5)
#     # print("generating R and P by selecting row...")
#     # R2 = scipy.sparse.csr_matrix((2,A.shape[0]), dtype=np.int32)
#     # R2[0,0] = 1
#     # R2[1,9] = 1
#     # P2 = R2.T
#     # x0 = np.zeros_like(b)
#     # _,residuals_selectRows = timer_wrapper(solve_rep_noSmoother, A, b, x0, R2, P2)

#     # print("generating R and P by removing rows...")
#     # R3 = scipy.sparse.identity(A.shape[0], dtype=np.int32)
#     # R3=R3.tocsr()
#     # R3 = delete_rows_csr(R3, range(0, A.shape[0] - 1, 2))
#     # P3 = R3.T
#     # print(f"##########R: {R3.shape}, P: {P3.shape}")
#     # x0 = np.zeros_like(b)
#     # print("rank of P3:", np.linalg.matrix_rank(P3.toarray()))
#     # _,residuals_removeRows = timer_wrapper(solve_rep_noSmoother, A, b, x0, R3, P3)

#     # ------------------------------- print results ---------------------------- #
#     # print("x_rep:", x_rep)
#     # x_rep_max = np.max(np.abs(x_rep))
#     # print("x_onlySmoother:", np.max(np.abs(x_rep-x_onlySmoother)/x_rep_max))
#     # print("x_noSmoother:", np.max(np.abs(x_rep-x_noSmoother)/x_rep_max))
#     # print("x_remove_offdiag:", np.max(np.abs(x_rep-x_remove_offdiag)/x_rep_max))
#     # print("x_reduce_offdiag:", np.max(np.abs(x_rep-x_reduce_offdiag)/x_rep_max))
#     # print("x_M_matrix:", np.max(np.abs(x_rep-x_M_matrix)/x_rep_max))


#     print_residuals(residuals_rep, "rep")
#     print_residuals(residuals_onlySmoother, "onlySmoother")
#     print_residuals(residuals_noSmoother, "noSmoother")
#     print_residuals(residuals_remove_offdiag, "remove_offdiag")
#     print_residuals(residuals_reduce_offdiag, "reduce_offdiag")
#     print_residuals(residuals_M_matrix, "M_matrix")

#     if show_plot:
#         fig, axs = plt.subplots(2, 1, figsize=(8, 9))
#         plot_residuals(residuals_rep, axs[0], label="rep")
#         plot_residuals(residuals_onlySmoother, axs[0], label="onlySmoother")
#         plot_residuals(residuals_remove_offdiag, axs[0],  label="remove_offdiag")
#         plot_residuals(residuals_reduce_offdiag, axs[0],  label="reduce_offdiag")
#         plot_residuals(residuals_M_matrix, axs[0],  label="M_matrix")
#         plot_residuals(residuals_noSmoother, axs[1],  label="noSmoother")

#         # plot_full_residual(full_residual_rep[0], "residual0")
#         # plot_full_residual(full_residual_rep[1], "residual1")
#         # plot_full_residual(full_residual_rep[2], "residual2")
#         # plot_full_residual(full_residual_rep[3], "residual3")

#         fig.canvas.manager.set_window_title(plot_title)
#         plt.tight_layout()
#         if save_fig_instad_of_show:
#             plt.savefig(f"result/latest/residuals_{plot_title}.png")
#         else:
#             plt.show()


# def test_amg2(A, b, postfix=''):
#     R1, P1 = generate_R_P(A)
#     x0 = np.zeros_like(b)
#     res_amg = []
#     x_amg = solve_amg(A, b, x0=np.zeros_like(b), R=R1, P=P1, residuals=res_amg, maxiter=maxiter)

#     res4 = []
#     x4=x0.copy()
#     for _ in range(maxiter+1):
#         x4 = gauss_seidel(A, x4, b, iterations=1, residuals=res4, tol=tol)
#         x4 = gauss_seidel(A, x4, b, iterations=1, residuals=res4, tol=tol)
#     print("res4 GS",len(res4), res4[-1])
#     print((res4[-1]/res4[0])**(1.0/(len(res4)-1)))


#     ml = pyamg.ruge_stuben_solver(A)
#     res1 = []
#     x_pyamg = ml.solve(b, tol=1e-10, residuals=res1,maxiter=maxiter)
#     print(ml)
#     print("res1 classical AMG", len(res1), res1[-1])
#     print((res1[-1]/res1[0])**(1.0/(len(res1)-1)))


#     if show_plot:
#         fig, axs = plt.subplots(1, 1, figsize=(8, 9))
#         plot_residuals(res_amg, axs, label="amg")
#         plot_residuals(res4, axs, label="gs2",marker='.')
#         plot_residuals(res1, axs, label="amg_full",marker='o')

#         fig.canvas.manager.set_window_title(plot_title)
#         plt.tight_layout()
#         if save_fig_instad_of_show:
#             plt.savefig(f"result/latest/residuals_{plot_title}.png")
#         else:
#             plt.show()




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

def SA_from_diagnostic(A, b, x0, res):
    # Generate B
    B = np.ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()
    # Create solver
    ml = pyamg.smoothed_aggregation_solver(A,B=B,BH=BH, 
        strength=('symmetric', {'theta': 0.0}),
        smooth="jacobi",
        improve_candidates=None,
        aggregate="standard",
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        max_levels=15,
        max_coarse=300,
        coarse_solver="pinv")
    x = ml.solve(b, x0=x0, tol=tol, residuals=res, accel="cg", maxiter=maxiter, cycle="W")


def load_A_b(case_name = "scale64"):
    import scipy.io
    import os
    import numpy as np
    postfix = "F1-0"

    print("loading data...")
    prj_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
    to_read_dir = prj_dir + f"result/{case_name}/A/"
    A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
    A = A.tocsr()
    print("shape of A: ", A.shape)
    b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float64)

    return A, b


def prepare_A_b(mat_size = 10, case_num = 0, postfix=""):
    global A, b

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
        A = A.astype(np.float64)
        b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float64)
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
        import concatenate_png
        if run_strength_options : prefix = 'strength'
        else: prefix = 'residuals'
        concatenate_png.concatenate_png(case_name, prefix)


if __name__ == "__main__":
    frames = [1, 6, 11, 16, 21, 26]
    for frame in frames:
        postfix=f"F{frame}-0"
        print(f"\n\n\n{postfix}")
        A,b = prepare_A_b(postfix=postfix)
        test_amg(A,b,postfix=postfix)

    import concatenate_png
    concatenate_png.concatenate_png(case_name, prefix='residuals', frames=frames)
