import numpy as np
import time
from time import perf_counter
import logging
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients

from engine.solver.build_Ps import build_Ps

class AmgPython:
    def __init__(self, args, get_A0, should_setup, copy_A=True):
        self.args = args
        self.get_A0 = get_A0
        self.should_setup = should_setup
        self.copy_A = copy_A

        self.Ps = None
        self.num_levels = None
        self.jacobi_omega = None
        self.chebyshev_coeff = None


    def run(self, b):
        logging.info("  AMG_python")
        A = self.get_A0()
        if self.copy_A:
            A = A.copy()#FIXME: softbody no copy will cause bug, but cloth is good, why?

        if self.should_setup():
            tic = time.perf_counter()
            self.Ps = build_Ps(A, self.args)
            self.num_levels = len(self.Ps) + 1
            logging.info(f"    build_Ps time:{time.perf_counter()-tic}")
        
        tic = time.perf_counter()
        levels = self.build_levels(A, self.Ps)
        logging.info(f"    build_levels time:{time.perf_counter()-tic}")

        if self.should_setup():
            tic = time.perf_counter()
            self.setup_smoothers(A)
            logging.info(f"    setup smoothers time:{perf_counter()-tic}")
        x0 = np.zeros_like(b)
        tic = time.perf_counter()
        x, r_Axb = self.old_amg_cg_solve(levels, b, x0=x0, maxiter=self.args.maxiter_Axb, tol=self.args.tol_Axb)
        toc = time.perf_counter()
        logging.info(f"    mgsolve time {toc-tic}")
        return  x, r_Axb


    # https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/relaxation/relaxation.py#L586
    def chebyshev(self, A, x, b):
        coefficients = self.chebyshev_coeff
        iterations = 1
        x = np.ravel(x)
        b = np.ravel(b)
        for _i in range(iterations):
            residual = b - A*x
            h = coefficients[0]*residual
            for c in coefficients[1:]:
                h = c*residual + A*h
            x += h

    def calc_spectral_radius(self, A):
        t = time.perf_counter()
        self.spectral_radius = approximate_spectral_radius(A) # legacy python version
        print(f"spectral_radius time: {time.perf_counter()-t:.2f}s")
        print("spectral_radius:",self.spectral_radius)
        return self.spectral_radius


    def setup_chebyshev(self, A):
        """Set up Chebyshev."""
        lower_bound=1.0/30.0
        upper_bound=1.1
        degree=3
        rho = self.calc_spectral_radius( A)
        a = rho * lower_bound
        b = rho * upper_bound
        self.chebyshev_coeff = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]


    def setup_jacobi(self, A):
        from pyamg.relaxation.smoothing import rho_D_inv_A
        rho = rho_D_inv_A(A)
        print("rho:", rho)
        self.jacobi_omega = 1.0/(rho)
        print("omega:", self.jacobi_omega)


    def setup_smoothers(self, A):
        if self.args.smoother_type == 'chebyshev':
            self.setup_chebyshev(A)
        elif self.args.smoother_type == 'jacobi':
            self.setup_jacobi(A)


    def build_levels(self, A, Ps=[]):
        class MultiLevel:
            A = None
            P = None
            R = None

        '''Give A and a list of prolongation matrices Ps, return a list of levels'''
        lvl = len(Ps) + 1 # number of levels

        levels = [MultiLevel() for i in range(lvl)]

        levels[0].A = A

        for i in range(lvl-1):
            levels[i].P = Ps[i]
            levels[i].R = Ps[i].T
            levels[i+1].A = Ps[i].T @ levels[i].A @ Ps[i]

        return levels


    def diag_sweep(self, A,x,b,iterations=1):
        diag = A.diagonal()
        diag = np.where(diag==0, 1, diag)
        x[:] = b / diag

    def presmoother(self, A,x,b):
        from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
        if self.args.smoother_type == 'gauss_seidel':
            gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
        elif self.args.smoother_type == 'jacobi':
            jacobi(A,x,b,iterations=10, omega=self.jacobi_omega)
        elif self.args.smoother_type == 'sor_vanek':
            for _ in range(1):
                sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
                sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
        elif self.args.smoother_type == 'sor':
            sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
        elif self.args.smoother_type == 'diag_sweep':
            self.diag_sweep(A,x,b,iterations=1)
        elif self.args.smoother_type == 'chebyshev':
            self.chebyshev(A,x,b)


    def postsmoother(self,A,x,b):
        self.presmoother(A,x,b)


    def coarse_solver(self, A, b):
        res = np.linalg.solve(A.toarray(), b)
        return res

    def old_V_cycle(self,levels,lvl,x,b):
        A = levels[lvl].A
        self.presmoother(A,x,b)
        residual = b - A @ x
        coarse_b = levels[lvl].R @ residual
        coarse_x = np.zeros_like(coarse_b)
        if lvl == len(levels)-2:
            coarse_x = self.coarse_solver(levels[lvl+1].A, coarse_b)
        else:
            self.old_V_cycle(levels, lvl+1, coarse_x, coarse_b)
        x += levels[lvl].P @ coarse_x
        self.postsmoother(A, x, b) 

    # non recursive V_cycle norecur
    def V_cycle_v2(self, levels, x0, b):
        nl = len(levels)
        levels[0].r=b
        levels[0].x=x0

        for l in range(nl - 1):
            A = levels[l].A
            levels[l].x = np.zeros(shape=A.shape[0])
            self.presmoother(A, levels[l].x, levels[l].r)
            levels[l+1].r = levels[l].R @ (levels[l].r - A @ levels[l].x)

        levels[nl-1].x = self.coarse_solver(levels[nl-1].A, levels[nl-1].r)

        for l in reversed(range(nl - 1)):
            levels[l].x += levels[l].P @ levels[l+1].x
            self.postsmoother(levels[l].A, levels[l].x, levels[l].r)

        return levels[0].x


    def old_amg_cg_solve(self, levels, b, x0=None, tol=1e-5, maxiter=100):
        assert x0 is not None
        x = x0.copy()
        A = levels[0].A
        residuals = np.zeros(maxiter+1)
        def psolve(b):
            x = x0.copy()
            self.old_V_cycle(levels, 0, x, b)
            return x
        bnrm2 = np.linalg.norm(b)
        atol = tol * bnrm2
        r = b - A@(x)
        rho_prev, p = None, None
        normr = np.linalg.norm(r)
        residuals[0] = normr
        iteration = 0
        for iteration in range(maxiter):
            if normr < atol:  # Are we done?
                break
            z = psolve(r)
            rho_cur = np.dot(r, z)
            if iteration > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z
            else:  # First spin
                p = np.empty_like(r)
                p[:] = z[:]
            q = A@(p)
            alpha = rho_cur / np.dot(p, q)
            x += alpha*p
            r -= alpha*q
            rho_prev = rho_cur
            normr = np.linalg.norm(r)
            residuals[iteration+1] = normr
        residuals = residuals[:iteration+1]
        return (x),  residuals  

