from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import scipy.io
import scipy.sparse
from time import perf_counter
from pathlib import Path

def load_A(path):
    print(f"loading data {path}...")
    path = Path(path)
    tic = perf_counter()
    if  path.suffix == ".npz":
        # https://stackoverflow.com/a/8980156/19253199
        A = scipy.sparse.load_npz(path)
        A = A.astype(np.float64)
        A = A.tocsr()
    elif path.suffix == ".mtx":
        A = scipy.io.mmread(path)
        A = A.tocsr()
        A = A.astype(np.float64)
    elif path.suffix == ".txt": #HACK for ruan2024
        with open(path,"r") as f:
            raw = np.loadtxt(path)
            [nrows, ncols, nnz] = raw[0].astype(int)
            raw = raw[1:]
            [rowidx, colidx, data] = raw[:,0].astype(int), raw[:,1].astype(int), raw[:,2].astype(float)
            A = coo_matrix((data.flatten(), (rowidx.flatten(), colidx.flatten())), shape=(nrows, ncols))
            A = A.tocsr()
            A = A.astype(np.float64)
    else:
        raise FileNotFoundError(f"Not in mtx or npz format")
    print(f"shape: {A.shape}, nnz: {A.nnz}")
    print(f"loading data {path} done in {perf_counter()-tic:.2f}s")
    return A