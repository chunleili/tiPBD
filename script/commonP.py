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
from utils.solvers import *

sys.path.append(os.getcwd())

save_fig = True
show_fig = False
show_time_plot = True
only_postprecess = False
run_concate_png = False

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
    commonP(A1, b,x0, allres, P0,P1)
    CAMG(A, b, x0, allres)
    CAMG_CG(A, b, x0, allres)

    df  = postprocess_residual_new(allres, tic)
    print_df_new(df)
    save_data_new(df,postfix)
    if show_time_plot:
        draw_times_new(df)
    plot_residuals_all_new(df,show_fig=show_fig,save_fig=save_fig,postfix=postfix, use_markers=True)


if __name__ == "__main__":

    frames = [1,6,11,16,21,26]
    ite = 0

    for frame in frames:
        postfix=f"F{frame}-{ite}"
        print(f"\n\n\n{postfix}")
        if only_postprecess:
            from script.utils.postprocess_residual import postprocess_from_file
            postprocess_from_file(postfix)
        else:
            A,b = load_A_b(postfix=postfix)
            test_amg(A,b,postfix=postfix)

    if run_concate_png:
        import script.utils.concatenate_png as concatenate_png
        concatenate_png.concatenate_png(case_name, imgs=[f"F{frame}-{ite}" for frame in frames])
