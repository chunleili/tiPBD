import numpy as np
import scipy.sparse.linalg
from time import perf_counter
import logging

class DirectSolver:
    def __init__(self, get_A0):
        self.get_A0 = get_A0
    
    
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