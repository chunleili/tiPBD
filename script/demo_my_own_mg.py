import os, sys
sys.path.append(os.getcwd())

from script.utils.amg_solve import amg_solve
from script.utils.load_A_b import load_A_b
from script.utils.build_Ps import build_Ps
from script.utils.build_levels import build_levels
from script.utils.plot_residuals import plot_residuals_all, draw_times
from script.utils.solvers import *

import numpy as np
from time import perf_counter
from collections import namedtuple
Residual = namedtuple('Residual', ['label','r', 't'])

def demo_my_own_mg():
    A, b = load_A_b('F11-0')
    t0= perf_counter()
    Ps = build_Ps(A)
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    print('Setup Time:', t1-t0)

    residuals = []
    x0 = np.zeros_like(b)
    t2= perf_counter()
    x = amg_solve(levels, b, x0=x0, maxiter=100, tol=1e-6, residuals=residuals)
    t3 = perf_counter()
    print('Solve Time:', t3-t2)
    print('Total Time:', t3-t2+t1-t0)
    allres = [Residual('MyOwnMG', residuals, t3-t0)]

    tic = perf_counter()
    UA_CG(A, b, x0, allres)
    toc = perf_counter()
    print("UA_CG Time:", toc-tic)

    plot_residuals_all(allres,use_markers=True)


if __name__ == "__main__":
    demo_my_own_mg()