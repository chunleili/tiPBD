# testLinSol.py

import scipy
from scipy.io import mmread, mmwrite
import numpy as np

A = mmread("D:/Dev/tiPBD/cpp/mgcg_cuda/lib/eigenA.mtx")
b = np.loadtxt("D:/Dev/tiPBD/cpp/mgcg_cuda/lib/b.mtx")

x = scipy.sparse.linalg.spsolve(A, b)
print(x)