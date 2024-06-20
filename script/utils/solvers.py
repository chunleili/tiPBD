import numpy as np
import scipy
import os, sys
from time import perf_counter
from matplotlib import pyplot as plt
import pyamg
from .parms import maxiter, tol, Residual
from .construct_ml_manually import construct_ml_manually_3levels


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


def commonP(A1, b, x0, allres, P0, P1):
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
