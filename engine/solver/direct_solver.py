import numpy as np
import scipy.sparse.linalg
from time import perf_counter
import logging

class DirectSolver:
    def __init__(self, get_A0=None):
        self.get_A0 = get_A0

    def run_cuda(self, A, b):
        def AMG_solve(self, b, x0=None, tol=1e-5, maxiter=100):
            if x0 is None:
                x0 = np.zeros(b.shape[0], dtype=np.float32)

            tic4 = perf_counter()
            # set data
            x0 = x0.astype(np.float32)
            b = b.astype(np.float32)
            self.extlib.fastmg_set_data(x0, x0.shape[0], b, b.shape[0], tol, maxiter)

            self.extlib.fastmg_solve_only_directsolver()

            # get result
            x = np.empty_like(x0, dtype=np.float32)
            residuals = np.zeros(shape=(maxiter,), dtype=np.float32)
            niter = self.extlib.fastmg_get_data(x, residuals)
            niter += 1
            residuals = residuals[:niter]
            logging.info(f"    inner iter: {niter}")
            logging.info(f"    solve time: {(perf_counter()-tic4)*1000:.0f}ms")
            logging.info(f"    residual: {residuals[0]:.6e} -> {residuals[-1]:.6e}")
            return (x),  residuals  

    
    def run2(self,A, b):
        tic = perf_counter()
        A = self.get_A0() # fill A
        A = A.copy() # we need copy for cloth, why?
        r_Axb = []
        r_Axb.append(np.linalg.norm(b))
        x = scipy.sparse.linalg.spsolve(A, b)
        if np.isnan(x).any():
            raise ValueError("DirectSolver: nan in x")
        r_Axb.append(np.linalg.norm(b-A@x))
        logging.info(f"    direct_solver time: {(perf_counter()-tic)*1000:.0f}ms")
        return x, r_Axb
    
    def run(self, b):
        # TODO: Better solution than scipy(cusolver)
        tic = perf_counter()
        A = self.get_A0() # fill A
        A = A.copy() # we need copy for cloth, why?
        r_Axb = []
        r_Axb.append(np.linalg.norm(b))
        x = scipy.sparse.linalg.spsolve(A, b)
        if np.isnan(x).any():
            raise ValueError("DirectSolver: nan in x")
        r_Axb.append(np.linalg.norm(b-A@x))
        logging.info(f"    direct_solver time: {(perf_counter()-tic)*1000:.0f}ms")
        return x, r_Axb