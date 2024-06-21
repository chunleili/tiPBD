from script.utils.build_Ps import build_Ps
from script.utils.build_levels import build_levels
from script.utils.amg_solve import amg_solve
import numpy as np

def setup_phase(A, method='UA_CG'):
    Ps = build_Ps(A, method=method)
    levels = build_levels(A, Ps)
    return levels

def solve_phase(levels, b, x0, maxiter, tol, residuals):
    x = amg_solve(levels, b, x0=x0, maxiter=maxiter, tol=tol, residuals=residuals)
    return x


def whole_amg_solver(A, b, x0, maxiter, tol, residuals, method='UA_CG'):
    levels = setup_phase(A, method='UA_CG')

    residuals = []
    x0 = np.zeros_like(b)
    x = solve_phase(levels, b, x0, maxiter=maxiter, tol=tol, residuals=residuals)
    return x