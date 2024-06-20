import numpy as np
from pyamg.relaxation.relaxation import gauss_seidel
from scipy.linalg import pinv

def presmoother(A,x,b):
    gauss_seidel(A,x,b,iterations=1, sweep='symmetric')

def postsmoother(A,x,b):
    gauss_seidel(A,x,b,iterations=1, sweep='symmetric')

def coarse_solver(A, b):
    P = pinv(A.toarray())
    return np.dot(P, b)

def V_cycle(levels,lvl,x,b):
    A = levels[lvl].A
    
    presmoother(A, x, b)
    
    residual = b - A @ x
    
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[-1].A, coarse_b)
    else:
        V_cycle(levels, lvl+1, coarse_x, coarse_b)

    x += levels[lvl].P @ coarse_x

    presmoother(A, x, b)
