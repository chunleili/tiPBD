import numpy as np
import scipy
import os, sys
from time import perf_counter
from .define_to_read_dir import to_read_dir
from pathlib import Path

def load_A_b(postfix):
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
        b = np.load(to_read_dir+f"b_{postfix}.npy")
        A = A.astype(np.float64)
        b = b.astype(np.float64)
        A = A.tocsr()
    else:
        A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
        A = A.tocsr()
        A = A.astype(np.float64)
        b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float64)
    print(f"loading data {postfix} done in {perf_counter()-tic:.2f}s")
    return A, b