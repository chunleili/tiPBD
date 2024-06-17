import numpy as np
from pyamg.util.utils import upcast, to_type
from pyamg.multilevel import coarse_grid_solver
from pyamg import krylov
import scipy.sparse.linalg as sla

coarse_solver = coarse_grid_solver('pinv')

def solve(levels,ml, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
            callback=None, residuals=None, cycles_per_level=1, return_info=False):
    """Execute multigrid cycling.

    Parameters
    ----------
    b : array
        Right hand side.
    x0 : array
        Initial guess.
    tol : float
        Stopping criteria: relative residual r[k]/||b|| tolerance.
        If `accel` is used, the stopping criteria is set by the Krylov method.
    maxiter : int
        Stopping criteria: maximum number of allowable iterations.
    cycle : {'V','W','F','AMLI'}
        Type of multigrid cycle to perform in each iteration.
    accel : string, function
        Defines acceleration method.  Can be a string such as 'cg'
        or 'gmres' which is the name of an iterative solver in
        pyamg.krylov (preferred) or scipy.sparse.linalg.
        If accel is not a string, it will be treated like a function
        with the same interface provided by the iterative solvers in SciPy.
    callback : function
        User-defined function called after each iteration.  It is
        called as callback(xk) where xk is the k-th iterate vector.
    residuals : list
        List to contain residual norms at each iteration.  The residuals
        will be the residuals from the Krylov iteration -- see the `accel`
        method to see verify whether this ||r|| or ||Mr|| (as in the case of
        GMRES).
    cycles_per_level: int, default 1
        Number of V-cycles on each level of an F-cycle
    return_info : bool
        If true, will return (x, info)
        If false, will return x (default)

    Returns
    -------
    x : array
        Approximate solution to Ax=b after k iterations

    info : string
        Halting status

        ==  =======================================
        0   successful exit
        >0  convergence to tolerance not achieved,
            return iteration count instead.
        ==  =======================================

    See Also
    --------
    aspreconditioner

    Examples
    --------
    >>> from numpy import ones
    >>> from pyamg import ruge_stuben_solver
    >>> from pyamg.gallery import poisson
    >>> A = poisson((100, 100), format='csr')
    >>> b = A * ones(A.shape[0])
    >>> ml = ruge_stuben_solver(A, max_coarse=10)
    >>> residuals = []
    >>> x = ml.solve(b, tol=1e-12, residuals=residuals) # standalone solver

    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0)  # copy

    A = levels[0].A

    cycle = str(cycle).upper()

    # AMLI cycles require hermitian matrix
    if (cycle == 'AMLI') and hasattr(A, 'symmetry'):
        if A.symmetry != 'hermitian':
            raise ValueError('AMLI cycles require \
                symmetry to be hermitian')

    if accel is not None:

        # # Check for symmetric smoothing scheme when using CG
        # if (accel == 'cg') and (not self.symmetric_smoothing):
        #     warn('Incompatible non-symmetric multigrid preconditioner '
        #             'detected, due to presmoother/postsmoother combination. '
        #             'CG requires SPD preconditioner, not just SPD matrix.')

        # Check for AMLI compatability
        if (accel != 'fgmres') and (cycle == 'AMLI'):
            raise ValueError('AMLI cycles require acceleration (accel) '
                                'to be fgmres, or no acceleration')

        # Acceleration is being used
        kwargs = {}
        if isinstance(accel, str):
            kwargs = {}
            if hasattr(krylov, accel):
                accel = getattr(krylov, accel)
            else:
                accel = getattr(sla, accel)
                kwargs['atol'] = 'legacy'

        M = ml.aspreconditioner(cycle=cycle)

        try:  # try PyAMG style interface which has a residuals parameter
            x, info = accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                            callback=callback, residuals=residuals, **kwargs)
            if return_info:
                return x, info
            return x
        except TypeError:
            # try the scipy.sparse.linalg style interface,
            # which requires a callback function if a residual
            # history is desired

            if residuals is not None:
                residuals[:] = [np.linalg.norm(b - A @ x)]

                def callback_wrapper(x):
                    if np.isscalar(x):
                        residuals.append(x)
                    else:
                        residuals.append(np.linalg.norm(b - A @ x))
                    if callback is not None:
                        callback(x)
            else:
                callback_wrapper = callback

            x, info = accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                            callback=callback_wrapper, **kwargs)
            if return_info:
                return x, info
            return x

    else:
        # Scale tol by normb
        # Don't scale tol earlier. The accel routine should also scale tol
        normb = np.linalg.norm(b)
        if normb == 0.0:
            normb = 1.0  # set so that we have an absolute tolerance

    # Start cycling (no acceleration)
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Create uniform types for A, x and b
    # Clearly, this logic doesn't handle the case of real A and complex b
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
            __solve(0, x, b, cycle, cycles_per_level)

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        if normr < tol * normb:
            if return_info:
                return x, 0
            return x

        if it == maxiter:
            if return_info:
                return x, it
            return x

def __solve(levels, lvl, x, b, cycle, cycles_per_level=1):
    """Multigrid cycling.

    Parameters
    ----------
    lvl : int
        Solve problem on level `lvl`
    x : numpy array
        Initial guess `x` and return correction
    b : numpy array
        Right-hand side for Ax=b
    cycle : {'V','W','F','AMLI'}
        Recursively called cycling function.  The
        Defines the cycling used:
        cycle = 'V',    V-cycle
        cycle = 'W',    W-cycle
        cycle = 'F',    F-cycle
        cycle = 'AMLI', AMLI-cycle
    cycles_per_level : int, default 1
        Number of V-cycles on each level of an F-cycle
    """
    A = levels[lvl].A

    levels[lvl].presmoother(A, x, b)

    residual = b - A @ x

    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    if lvl == len(levels) - 2:
        coarse_x[:] = coarse_solver(levels[-1].A, coarse_b)
    else:
        if cycle == 'V':
            __solve(lvl + 1, coarse_x, coarse_b, 'V')
        elif cycle == 'W':
            __solve(lvl + 1, coarse_x, coarse_b, cycle)
            __solve(lvl + 1, coarse_x, coarse_b, cycle)
        elif cycle == 'F':
            __solve(lvl + 1, coarse_x, coarse_b, cycle, cycles_per_level)
            for _ in range(0, cycles_per_level):
                __solve(lvl + 1, coarse_x, coarse_b, 'V', 1)
        elif cycle == 'AMLI':
            # Run nAMLI AMLI cycles, which compute "optimal" corrections by
            # orthogonalizing the coarse-grid corrections in the A-norm
            nAMLI = 2
            Ac = levels[lvl + 1].A
            p = np.zeros((nAMLI, coarse_b.shape[0]), dtype=coarse_b.dtype)
            beta = np.zeros((nAMLI, nAMLI), dtype=coarse_b.dtype)
            for k in range(nAMLI):
                # New search direction --> M^{-1}*residual
                p[k, :] = 1
                __solve(lvl + 1, p[k, :].reshape(coarse_b.shape),
                                coarse_b, cycle)

                # Orthogonalize new search direction to old directions
                for j in range(k):  # loops from j = 0...(k-1)
                    beta[k, j] = np.inner(p[j, :].conj(), Ac * p[k, :]) /\
                        np.inner(p[j, :].conj(), Ac * p[j, :])
                    p[k, :] -= beta[k, j] * p[j, :]

                # Compute step size
                Ap = Ac * p[k, :]
                alpha = np.inner(p[k, :].conj(), np.ravel(coarse_b)) /\
                    np.inner(p[k, :].conj(), Ap)

                # Update solution
                coarse_x += alpha * p[k, :].reshape(coarse_x.shape)

                # Update residual
                coarse_b -= alpha * Ap.reshape(coarse_b.shape)
        else:
            raise TypeError(f'Unrecognized cycle type ({cycle})')

    x += levels[lvl].P @ coarse_x   # coarse grid correction

    levels[lvl].postsmoother(A, x, b)


