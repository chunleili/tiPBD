from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse
from time import perf_counter
from pathlib import Path

to_read_dir = "./result/case98-1019-soft85w/A/"


def load_A(postfix):
    print(f"loading data {postfix} in {to_read_dir}...")
    path = to_read_dir+f"A_{postfix}"
    if  Path(path+".npz").exists():
        binary = True
    elif Path(path+".mtx").exists():
        binary = False
    else:
        raise FileNotFoundError(f"File not found: {path}")
    tic = perf_counter()
    if binary:
        # https://stackoverflow.com/a/8980156/19253199
        A = scipy.sparse.load_npz(to_read_dir+f"A_{postfix}.npz")
        A = A.astype(np.float64)
        A = A.tocsr()
    else:
        A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
        A = A.tocsr()
        A = A.astype(np.float64)
    print(f"loading data {postfix} done in {perf_counter()-tic:.2f}s")
    return A



fig, axs = plt.subplots(3,figsize=(5, 10))

A0 = load_A("L0")
print("A:", A0.shape)
print("nnz:", A0.nnz)
axs[0].spy(A0, markersize=5e-7, markevery=500)#L0

A = load_A("L1")
print("A:", A.shape)
print("nnz:", A.nnz)
axs[1].spy(A, markersize=1e-5, markevery=5)#L1

A = load_A("L2")
print("A:", A.shape)
print("nnz:", A.nnz)
axs[2].spy(A, markersize=1, markevery=1)#L2

# plt.tight_layout()
axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
axs[2].ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.show()