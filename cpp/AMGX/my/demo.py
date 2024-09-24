import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import os

os.add_dll_directory("D:/Dev/AMGX/build/Release")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")

dir = os.path.dirname(os.path.abspath(__file__))

import pyamgx

pyamgx.initialize()

cfg = pyamgx.Config().create_from_file(dir+"/configs/AGGREGATION_JACOBI.json")

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

# Upload system:
M = sparse.csr_matrix(np.random.rand(5, 5))
rhs = np.random.rand(5)
sol = np.zeros(5, dtype=np.float64)

A.upload_CSR(M)
b.upload(rhs)
x.upload(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)

# Download solution
x.download(sol)
print("pyamgx solution: ", sol)
print("scipy solution: ", splinalg.spsolve(M, rhs))

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()
