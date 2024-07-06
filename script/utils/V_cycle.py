import numpy as np
from pyamg.relaxation.relaxation import gauss_seidel
from time import perf_counter
from .coarse_solver import coarse_solver
from .smoothers import presmoother, postsmoother

def V_cycle(levels,lvl,x,b):
    A = levels[lvl].A
    
    presmoother(A, x, b)
    
    residual = b - A @ x
    
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        V_cycle(levels, lvl+1, coarse_x, coarse_b)

    x += levels[lvl].P @ coarse_x

    postsmoother(A, x, b)


# non recursive V_cycle
def V_cycle2(levels, x0, b):
    nl = len(levels)
    levels[0].r=b
    levels[0].x=x0

    for l in range(nl - 1):
        A = levels[l].A
        levels[l].x = np.zeros(shape=A.shape[0])
        presmoother(A, levels[l].x, levels[l].r)
        levels[l+1].r = levels[l].R @ (levels[l].r - A @ levels[l].x)

    levels[nl-1].x = coarse_solver(levels[nl-1].A, levels[nl-1].r)

    for l in reversed(range(nl - 1)):
        levels[l].x += levels[l].P @ levels[l+1].x
        postsmoother(levels[l].A, levels[l].x, levels[l].r)

    return levels[0].x
