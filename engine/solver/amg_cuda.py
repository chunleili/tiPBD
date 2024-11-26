import logging
import numpy as np
import time
import ctypes
import scipy

class AmgCuda:
    """
    AmgCuda 类用于在 CUDA 上运行 AMG 算法。

    Examples
    --------
    see test_amg_cuda()
    """
    def __init__(
        self,
        args,
        extlib,
        get_A0=None,
        fill_A_in_cuda=None,
        should_setup=None,
        graph_coloring=None,
        copy_A=True,
        only_smoother=None,
        only_jacobi=None,
        only_direct=None,
    ):
        """
        Initialize an instance of the AmgCuda class.

        Parameters
        ----------
        args : Command line arguments or configuration object.

        extlib : External library object for calling CUDA functions.

        get_A0 : Function pointer for obtaining the matrix A0.

        fill_A_in_cuda : Function pointer for filling the matrix on the CUDA side.

        should_setup : Function pointer to determine if setup is needed.

        graph_coloring : Graph coloring object (optional).

        copy_A : Boolean indicating whether to copy the matrix A (default is True).
        """
        self.args = args
        self.extlib = extlib
        self.copy_A = copy_A

        # TODO: for now, we pass func ptr to distinguish between soft and cloth
        self.get_A0 = get_A0
        self.fill_A_in_cuda = fill_A_in_cuda
        self.graph_coloring = graph_coloring
        self.should_setup = should_setup
        self.only_smoother = only_smoother
        self.only_jacobi = only_jacobi
        self.only_direct = only_direct

        if self.should_setup is None:
            self.should_setup = lambda: True #always setup
        if self.fill_A_in_cuda is None:
            self.fill_A_in_cuda = get_A0 #default to get_A0
        if self.only_smoother is None:
            self.only_smoother = self.args.only_smoother
        if self.only_jacobi is None:
            self.only_jacobi = False
        if self.only_direct is None:
            self.only_direct = False


    def run(self, b):
        if self.should_setup():
            A = self.AMG_setup_phase()
        self.fill_A_in_cuda()
        # self.AMG_RAP() # will call in solve
        x, r_Axb = self.AMG_solve(b, maxiter=self.args.maxiter_Axb, tol=self.args.tol_Axb)
        return x, r_Axb

    def AMG_solve(self, b, x0=None, tol=1e-5, maxiter=100):
        if x0 is None:
            x0 = np.zeros(b.shape[0], dtype=np.float32)

        tic4 = time.perf_counter()
        # set data
        x0 = x0.astype(np.float32)
        b = b.astype(np.float32)
        self.extlib.fastmg_set_data(x0, x0.shape[0], b, b.shape[0], tol, maxiter)

        # solve
        if self.only_smoother:
            self.extlib.fastmg_solve_only_smoother()
        elif self.only_jacobi:
            self.extlib.fastmg_solve_only_jacobi()
        elif self.only_direct:
            self.extlib.fastmg_solve_only_directsolver()
        else:
            self.extlib.fastmg_solve()

        # get result
        x = np.empty_like(x0, dtype=np.float32)
        residuals = np.zeros(shape=(maxiter,), dtype=np.float32)
        niter = self.extlib.fastmg_get_data(x, residuals)
        niter += 1
        residuals = residuals[:niter]
        logging.info(f"    inner iter: {niter}")
        logging.info(f"    solve time: {(time.perf_counter()-tic4)*1000:.0f}ms")
        logging.info(f"    residual: {residuals[0]:.6e} -> {residuals[-1]:.6e}")
        return (x),  residuals  

    def AMG_RAP(self):
        tic3 = time.perf_counter()
        for lv in range(self.num_levels-1):
            self.extlib.fastmg_RAP(lv) 
        logging.info(f"    RAP time: {(time.perf_counter()-tic3)*1000:.0f}ms")

    def update_P(self,Ps):
        for lv in range(len(Ps)):
            P_ = (Ps[lv]).tocsr()
            self.extlib.fastmg_set_P(lv, P_.data.astype(np.float32), P_.indices, P_.indptr, P_.shape[0], P_.shape[1], P_.nnz)

    def cuda_set_A0(self,A0):
        self.extlib.fastmg_set_A0(A0.data.astype(np.float32), A0.indices, A0.indptr, A0.shape[0], A0.shape[1], A0.nnz)

    def AMG_setup_phase(self, A=None):
        if self.only_direct:
            return None
        
        tic = time.perf_counter()
        if A is None:
            A = self.get_A0()
            if self.copy_A:
                A = A.copy() #FIXME: no copy will cause bug, why?
            A = A.tocsr().astype(np.float32)

        if self.only_smoother:
            self.cuda_set_A0(A)
            self.setup_smoothers()
            return A
        if self.only_jacobi:
            self.cuda_set_A0(A)
            self.args.smoother_type = "jacobi"
            self.args.smoother_niter = 1
            self.setup_smoothers()
            return A
        
        from engine.solver.build_Ps import build_Ps
        self.Ps = build_Ps(A, self.args, self.extlib)
        self.num_levels = len(self.Ps)+1
        logging.info(f"    build_Ps time:{time.perf_counter()-tic}")

        if self.num_levels == 1:
            # fallback to smoother only
            self.cuda_set_A0(A)
            self.setup_smoothers()
            self.only_smoother = True


        tic = time.perf_counter()
        self.update_P(self.Ps)
        logging.info(f"    update_P time: {time.perf_counter()-tic:.2f}s")

        tic = time.perf_counter()
        self.cuda_set_A0(A)

        self.AMG_RAP()

        self.setup_smoothers()
        self.extlib.fastmg_set_coarse_solver_type.argtypes = [ctypes.c_int]
        self.extlib.fastmg_set_coarse_solver_type(self.args.coarse_solver_type)

        logging.info(f"    setup smoothers time:{time.perf_counter()-tic}")
        return A

    def setup_smoothers(self):
        s = smoother_name2type(self.args.smoother_type)
        c_int = ctypes.c_int
        self.extlib.fastmg_setup_smoothers.argtypes = [c_int]
        self.extlib.fastmg_setup_smoothers(s) # 1 means chebyshev, 2 means w-jacobi, 3 gauss_seidel
        self.extlib.fastmg_set_smoother_niter(self.args.smoother_niter)
        if self.args.smoother_type=="gauss_seidel":
            self.graph_coloring()    



    def run_v2(self, A, b):
        def AMG_setup_phase_v2(self, A=None):
            if self.only_direct:
                return None
            
            tic = time.perf_counter()

            if self.only_smoother:
                self.cuda_set_A0(A)
                self.setup_smoothers()
                return A
            if self.only_jacobi:
                self.cuda_set_A0(A)
                self.args.smoother_type = "jacobi"
                self.args.smoother_niter = 1
                self.setup_smoothers()
                return A
            
            from engine.solver.build_Ps import build_Ps
            self.Ps = build_Ps(A, self.args, self.extlib)
            self.num_levels = len(self.Ps)+1
            logging.info(f"    build_Ps time:{time.perf_counter()-tic}")

            if self.num_levels == 1:
                # fallback to smoother only
                self.cuda_set_A0(A)
                self.setup_smoothers()
                self.only_smoother = True

            tic = time.perf_counter()
            self.update_P(self.Ps)
            logging.info(f"    update_P time: {time.perf_counter()-tic:.2f}s")

            tic = time.perf_counter()
            self.cuda_set_A0(A)

            self.AMG_RAP()

            self.setup_smoothers()
            self.extlib.fastmg_set_coarse_solver_type.argtypes = [ctypes.c_int]
            self.extlib.fastmg_set_coarse_solver_type(self.args.coarse_solver_type)

            logging.info(f"    setup smoothers time:{time.perf_counter()-tic}")
            return A
        

        def AMG_solve_v2(self,A, b, x0=None, tol=1e-5, maxiter=100):
            if x0 is None:
                x0 = np.zeros(b.shape[0], dtype=np.float32)

            tic4 = time.perf_counter()
            # set data
            x0 = x0.astype(np.float32)
            b = b.astype(np.float32)
            self.extlib.fastmg_set_data(x0, x0.shape[0], b, b.shape[0], tol, maxiter)

            self.cuda_set_A0(A)

            # solve
            if self.only_smoother:
                self.extlib.fastmg_solve_only_smoother()
            elif self.only_jacobi:
                self.extlib.fastmg_solve_only_jacobi()
            elif self.only_direct:
                self.extlib.fastmg_solve_only_directsolver()
            else:
                self.extlib.fastmg_solve()

            # get result
            x = np.empty_like(x0, dtype=np.float32)
            residuals = np.zeros(shape=(maxiter,), dtype=np.float32)
            niter = self.extlib.fastmg_get_data(x, residuals)
            niter += 1
            residuals = residuals[:niter]
            logging.info(f"    inner iter: {niter}")
            logging.info(f"    solve time: {(time.perf_counter()-tic4)*1000:.0f}ms")
            logging.info(f"    residual: {residuals[0]:.6e} -> {residuals[-1]:.6e}")
            return (x),  residuals  
            
        if self.should_setup():
            AMG_setup_phase_v2(self, A)
        x, r_Axb = AMG_solve_v2(self, A, b, maxiter=self.args.maxiter_Axb, tol=self.args.tol_Axb)
        return x, r_Axb



def smoother_name2type(name):
    if name == "chebyshev":
        return 1
    elif name == "jacobi":
        return 2
    elif name == "gauss_seidel":
        return 3
    else:
        raise ValueError(f"smoother name {name} not supported")
