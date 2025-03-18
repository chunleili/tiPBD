import numpy as np
from time import perf_counter
import logging

from engine.linearSystem.sparse_gauss_seidel import sparse_gauss_seidel_kernel

class GaussSeidelSolver:
    def __init__(self, get_A0, calc_residual_every_iter=False):
        self.get_A0 = get_A0
        self.calc_residual_every_iter = calc_residual_every_iter
    
    def run(self, b, x0=None, maxiter=1):
        tic = perf_counter()
        A = self.get_A0() # fill A
        A = A.copy() # we need copy for cloth, why?
        r_Axb = []
        r_Axb.append(np.linalg.norm(b))
        if x0 is None:
            x0 = np.zeros_like(b)
        x = x0.copy()
        logging.info(f"    gauss_seidel maxiter: {maxiter}")
        for _ in range(maxiter):
            sparse_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
            if self.calc_residual_every_iter:
                r_Axb.append(np.linalg.norm(b-A@x))
        if np.isnan(x).any():
            raise ValueError("nan in x")
        if not self.calc_residual_every_iter:
            r_Axb.append(np.linalg.norm(b-A@x))
        logging.info(f"    gauss_seidel time: {(perf_counter()-tic)*1000:.0f}ms")
        return x, r_Axb
