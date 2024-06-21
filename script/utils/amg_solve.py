import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from pyamg.util.utils import to_type
from scipy.sparse.sputils import upcast
from .V_cycle import V_cycle
from time import perf_counter
from .coarse_solver import coarse_solver

def aspreconditioner(levels, x0):
    A0 = levels[0].A
    shape = A0.shape
    dtype = A0.dtype

    def matvec(b):
        return amg_standalone_solve(levels, b, x0, maxiter=1, tol=1e-12)

    return LinearOperator(shape, matvec, dtype=dtype)


def amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100,residuals=None, return_info=False):
    x = x0.copy()
    A = levels[0].A

    if residuals is not None:
        residuals[:] = [np.linalg.norm(b - A @ x)]
        def callback_wrapper(x):
            if np.isscalar(x):
                residuals.append(x)
            else:
                residuals.append(np.linalg.norm(b - A @ x))

    M = aspreconditioner(levels, x0)

    x, info = cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M, callback=callback_wrapper)
    if return_info:
        return x, info
    return x


def amg_standalone_solve(levels, b, x0=None, tol=1e-5, maxiter=100,
              residuals=None, return_info=False):
    x = x0.copy()
    A = levels[0].A

    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance

    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual
    

    tp = upcast(b.dtype, x.dtype, A.dtype)
    [b, x] = to_type(tp, [b, x])
    b = np.ravel(b)
    x = np.ravel(x)

    it = 0
    
    while True:  # it <= maxiter and normr >= tol:
        if len(levels) == 1:
            # hierarchy has only 1 level
            x = coarse_solver(A, b)
        else:
            V_cycle(levels, 0, x, b)

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)

        if normr < tol * normb:
            if return_info:
                return x, 0
            return x

        if it == maxiter:
            if return_info:
                return x, it
            return x
        

# Only one iteration amg, for amg_cg to call. We dont need to check the residual,
# which will save some time.
def amg_standalone_solve_once(levels, b, x0):
    x = x0.copy()
    A = levels[0].A

    if len(levels) == 1:
        # hierarchy has only 1 level
        x = levels[-1].coarse_solver(A, b)
    else:
        V_cycle(levels, 0, x, b)
        
    toc = perf_counter()
    return x


amg_solve = amg_cg_solve