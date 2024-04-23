"""Build P based on Ruge-Stueben algorithm"""
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

A = poisson((20,))
print(A.toarray())
ml = pyamg.ruge_stuben_solver(A, max_levels=2)
P = ml.levels[0].P
R = ml.levels[0].R
r = []
x = ml.solve(np.ones(A.shape[0]), tol=1e-3, residuals=r, maxiter=1)
print(P.toarray())
print(x)
print(r)