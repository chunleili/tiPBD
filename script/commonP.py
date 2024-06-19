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
from utils.define_to_read_dir import case_name, to_read_dir
from utils.load_A_b import load_A_b
from utils.construct_ml_manually import construct_ml_manually_3levels
from utils.plot_residuals import plot_residuals_all_new,  draw_times_new
from utils.postprocess_residual import print_df_new, save_data_new, postprocess_residual_new

sys.path.append(os.getcwd())


save_fig = True
show_fig = False
show_time_plot = True
maxiter = 300
early_stop = False
tol=1e-10 # relative tolerance
run_concate_png = True
postfix = ''
only_postprecess = False

Residual = namedtuple('Residual', ['label','r', 't'])

A1 = load_A_b("F1-0")[0]


def test_amg(A, b, postfix=""):
    # x0 = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    allres = []
    tic = perf_counter()

    GS(A, b,x0, allres)
    CG(A, b,x0, allres,)
    ml17=SA_CG(A, b,x0, allres)
    P0 = ml17.levels[0].P
    P1 = ml17.levels[1].P
    commonP(b,x0, allres, P0,P1)
    injectionP(A, b, x0, allres)

    df  = postprocess_residual_new(allres, tic)
    print_df_new(df)
    save_data_new(df,postfix)
    if show_time_plot:
        draw_times_new(df)
    plot_residuals_all_new(df,show_fig=show_fig,save_fig=save_fig,postfix=postfix, use_markers=True)


def injectionP(A,b,x0, allres):
    label = "Classical injectionP"
    print(f"Calculating {label}...")
    ml19 = pyamg.ruge_stuben_solver(A, max_coarse=400, keep=True, interpolation='injection')
    r = []
    _ = ml19.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    allres.append(Residual(label, r, perf_counter()))
    print("len(level)=", len(ml19.levels))


def GS(A,b,x0,allres):
    label = "GS"
    print(f"Calculating {label}...")
    x4 = x0.copy()
    r = []
    for _ in range(maxiter+1):
        r.append(np.linalg.norm(b - A @ x4))
        pyamg.relaxation.relaxation.gauss_seidel(A=A, x=x4, b=b, iterations=1)
    allres.append(Residual(label, r, perf_counter()))


def commonP(b,x0, allres, P0, P1):
    label = "commonP A1"
    print(f"Calculating {label}...")
    tt1 = perf_counter()
    ml18 = construct_ml_manually_3levels(A1,P0,P1)
    print("setup phase of commonP time=", perf_counter()-tt1)
    r = []
    tt = perf_counter()
    _ = ml18.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter, accel='cg')
    print("solve phase of common P time=", perf_counter()-tt)
    allres.append(Residual(label, r, perf_counter()))


def SA_CG(A, b, x0, allres):
    label = "SA+CG"
    print(f"Calculating {label}...")
    tt = perf_counter()
    ml17 = pyamg.smoothed_aggregation_solver(A, max_coarse=400, keep=True)
    print("setup phase of SA time=", perf_counter()-tt)
    r = []
    tt = perf_counter()
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    print("solve phase of SA time=", perf_counter()-tt)
    allres.append(Residual(label, r, perf_counter()))
    print("len(level)=", len(ml17.levels))
    return ml17


def CG(A, b, x0, allres):
    label = "CG"
    print(f"Calculating {label}...")
    x6 = x0.copy()
    r = []
    r.append(np.linalg.norm(b - A @ x6))
    x6 = scipy.sparse.linalg.cg(A, b, x0=x0.copy(), rtol=tol, maxiter=maxiter, callback=lambda x: r.append(np.linalg.norm(b - A @ x)))
    allres.append(Residual(label, r, perf_counter()))


if __name__ == "__main__":
    frames = [1,6,11,16,21,26]
    ite = 0

    if only_postprecess:
        from script.utils.postprocess_residual import postprocess_from_file

        for frame in frames:
            postfix=f"F{frame}-{ite}"
            print(f"\n\n\n{postfix}")
            postprocess_from_file(postfix)

    else:
        for frame in frames:
            postfix=f"F{frame}-{ite}"
            print(f"\n\n\n{postfix}")
            A,b = load_A_b(postfix=postfix)
            test_amg(A,b,postfix=postfix)

    import script.utils.concatenate_png as concatenate_png
    concatenate_png.concatenate_png(case_name, imgs=[f"F{frame}-{ite}" for frame in frames])
