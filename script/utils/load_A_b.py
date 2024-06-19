import numpy as np
import scipy
import os, sys
from utils.define_to_read_dir import to_read_dir

def load_A_b(postfix):
    print(f"loading data {postfix}...")
    A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
    A = A.tocsr()
    A = A.astype(np.float64)
    b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float64)
    return A, b