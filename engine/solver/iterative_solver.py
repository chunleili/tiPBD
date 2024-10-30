import numpy as np
from time import perf_counter
import logging

from engine.linearSystem.sparse_gauss_seidel import sparse_gauss_seidel_kernel

class GaussSeidelSolver:
    def __init__(self, get_A0):
        self.get_A0 = get_A0
    
    def run(self, b, maxiter_Axb=1):
        tic = perf_counter()
        A = self.get_A0() # fill A
        A = A.copy() # we need copy for cloth, why?
        r_Axb = []
        r_Axb.append(np.linalg.norm(b))
        x = np.zeros_like(b)
        x0 = x.copy()
        logging.info(f"    gauss_seidel maxiter_Axb: {maxiter_Axb}")
        for _ in range(maxiter_Axb):
            sparse_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
        if np.isnan(x).any():
            raise ValueError("nan in x")
        r_Axb.append(np.linalg.norm(b-A@x))
        logging.info(f"    gauss_seidel time: {(perf_counter()-tic)*1000:.0f}ms")
        return x, r_Axb
