"""测试不同参数的AMG找到最佳收敛和最快速度"""
import numpy as np
import scipy
import os, sys
from time import perf_counter
from matplotlib import pyplot as plt
import pyamg
from pyamg.relaxation.smoothing import change_smoothers
from collections import namedtuple
import argparse

sys.path.append(os.getcwd())

prj_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("prj_dir", prj_dir)

parser = argparse.ArgumentParser()
parser.add_argument("-case_name", type=str, default='latest')
case_name = parser.parse_args().case_name

to_read_dir = prj_dir + f"result/{case_name}/A/"
save_fig = True
show_fig = True
maxiter = 150
early_stop = False
tol=1e-10 # relative tolerance
run_concate_png = True
postfix = ''

Residual = namedtuple('Residual', ['label','r', 't'])

def test_amg(A, b, postfix=""):
    # x0 = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    allres = []
    tic = perf_counter()

    label = "GS"
    print(f"Calculating {label}...")
    x4 = x0.copy()
    r = []
    for _ in range(maxiter+1):
        r.append(np.linalg.norm(b - A @ x4))
        pyamg.relaxation.relaxation.gauss_seidel(A=A, x=x4, b=b, iterations=1)
    allres.append(Residual(label, r, perf_counter()))



    # CG
    label = "CG"
    print(f"Calculating {label}...")
    x6 = x0.copy()
    r = []
    r.append(np.linalg.norm(b - A @ x6))
    x6 = scipy.sparse.linalg.cg(A, b, x0=x0.copy(), rtol=tol, maxiter=maxiter, callback=lambda x: r.append(np.linalg.norm(b - A @ x)))
    allres.append(Residual(label, r, perf_counter()))



    label = "SA+CG"
    print(f"Calculating {label}...")
    ml17 = pyamg.smoothed_aggregation_solver(A, max_coarse=400, keep=True)
    r = []
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))
    print("len(level)=", len(ml17.levels))

    fig, ax = plt.subplots(3, figsize=(8, 9))
    ax[0].spy(ml17.levels[0].A, markersize=1)
    ax[1].spy(ml17.levels[1].A, markersize=1)
    sparsity = ml17.levels[1].A.nnz/(ml17.levels[1].A.shape[0] * ml17.levels[1].A.shape[1])
    print(f"sparsity of level 1: {sparsity}")
    ax[2].spy(ml17.levels[2].A, markersize=1)
    plt.show()



    label = "UA+CG"
    print(f"Calculating {label}...")
    ml19 = pyamg.smoothed_aggregation_solver(A, smooth=None, max_coarse=400, keep=True)
    r = []
    _ = ml19.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))
    print("len(level)=", len(ml19.levels))

    fig, ax = plt.subplots(3, figsize=(8, 9))
    ax[0].spy(ml19.levels[0].A, markersize=1)
    sparsity = ml19.levels[1].A.nnz/(ml19.levels[1].A.shape[0] * ml19.levels[1].A.shape[1])
    print(f"sparsity of level 1: {sparsity}")
    ax[1].spy(ml19.levels[1].A, markersize=1)
    ax[2].spy(ml19.levels[2].A, markersize=1)
    plt.show()


    label = "SAgg+CG(no inject)"
    print(f"Calculating {label}...")
    ml18 = pyamg.smoothed_aggregation_solver(A, smooth=None, max_coarse=400,keep=True)

    Agg0 = ml18.levels[0].AggOp
    Agg1 = ml18.levels[1].AggOp
    from pyamg.util.utils import levelize_smooth_or_improve_candidates
    smooth = ['jacobi']
    smooth = levelize_smooth_or_improve_candidates(smooth, 2)
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}
    from pyamg.aggregation.smooth import jacobi_prolongation_smoother

    for i in range(len(ml18.levels)-1):
        T = ml18.levels[i].T
        C = ml18.levels[i].C
        B = ml18.levels[i].B
        # fn, kwargs = unpack_arg(smooth[len(levels)-1])

        from pyamg.util.utils import scale_rows, get_diagonal, get_block_diag, \
    unamal, filter_operator, compute_BtBinv, filter_matrix_rows, \
    truncate_rows
        from pyamg.util.linalg import approximate_spectral_radius
        # Use diagonal of S
        omega = 4./3.
        S = C
        D_inv = get_diagonal(S, inv=True)
        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = (omega/approximate_spectral_radius(D_inv_S))*D_inv_S

        # Carry out Jacobi as normal
        P = T
        for _ in range(1):
            P = P - (D_inv_S*P)

        ml18.levels[i].P = P


    r = []
    _ = ml18.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))
    print("len(level)=", len(ml18.levels))
    








    convs,times,labels  = postprocess_residual(allres, tic)
    
    # draw_convergence_factors(convs, labels)
    # draw_times(times, labels)

    df = print_df(labels, convs, times)
    save_data(allres,postfix)

    # draw_plot
    colors = ['blue', 'orange', 'red', 'purple', 'green', 'black', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'brown', 'pink']
    markers = ['o', 'x', 's', 'd', '^', 'v', '>', '<', '1', '2', '3', '4', '+', 'X']
    markers = ['' for _ in range(len(allres)-1)]
    # markers.append('o')

    # https://matplotlib.org/stable/api/markers_api.html for different markers
    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def for different colors
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    fig, axs = plt.subplots(1, figsize=(8, 9))
    for i in range(len(allres)):
        # if allres[i].label == 'SA+CG' or\
        #    allres[i].label == 'UA+CG' or\
        #    allres[i].label == 'GS':
        plot_residuals(a2r(allres[i].r), axs,  label=allres[i].label)

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



def load_A_b(postfix):
    print("loading data...")
    A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
    A = A.tocsr()
    A = A.astype(np.float64)
    b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float64)
    return A, b


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


if __name__ == "__main__":
    frames = [1, 6, 11, 16, 21, 26]
    for frame in frames:
        postfix=f"F{frame}-0"
        print(f"\n\n\n{postfix}")
        A,b = load_A_b(postfix=postfix)
        test_amg(A,b,postfix=postfix)

        postfix=f"F{frame}-1"
        print(f"\n\n\n{postfix}")
        # A,b = load_A_b(postfix=postfix)
        test_amg(A,b,postfix=postfix)

    import script.utils.concatenate_png as concatenate_png
    concatenate_png.concatenate_png(case_name, prefix='residuals', frames=frames)
