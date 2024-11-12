import numpy as np
import scipy.sparse as sparse
import os
import time
import logging

class AmgxSolver:
    '''
        usage: see test_amgx()
    '''
    def __init__(self, config_file, get_A0, cuda_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin", amgx_lib_dir="D:/Dev/AMGX/build/Release"):
        self.config_file = config_file
        # amgx_lib_dir = "D:/Dev/AMGX/build/Release"
        # cuda_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
        os.add_dll_directory(amgx_lib_dir)
        os.add_dll_directory(cuda_dir)
        import pyamgx
        self.pyamgx = pyamgx

        self.get_A0 = get_A0
        self.init()

    def run(self, b):
        tic = time.perf_counter()
        A = self.get_A0()
        A = A.copy()#FIXME: no copy will cause bug, why?
        # x, r_Axb, niter = AmgxSolver.update(A.data, b) TODO: fix this
        x, r_Axb, niter = self.solve(A, b)
        # print(f"    AMGX residual {r_Axb[:]}")
        # print(f"    AMGX niter: {niter}")
        logging.info(f"    AMGX time: {(time.perf_counter()-tic)*1000:.0f}ms")
        return x, np.array(r_Axb)
    

    def init(self):
        self.pyamgx.initialize()
        self.cfg = self.pyamgx.Config().create_from_file(self.config_file)
        self.rsc = self.pyamgx.Resources().create_simple(self.cfg)
        # Create matrices and vectors:
        self.A = self.pyamgx.Matrix().create(self.rsc)
        self.b = self.pyamgx.Vector().create(self.rsc)
        self.x = self.pyamgx.Vector().create(self.rsc)
        # Create solver:
        self.solver = self.pyamgx.Solver().create(self.rsc, self.cfg)


    def update(self, data, rhs):
        self.A.replace_coefficients(data.astype(np.float64))
        self.b.upload(rhs.astype(np.float64))
        
        self.solver.setup(self.A)
        self.solver.solve(self.b, self.x)
        self.niter = self.solver.iterations_number
        self.status = self.solver.status

        # assert self.status == 'success'
        logging.info("pyamgx status: ", self.status)
        self.r_Axb = []
        for i in range(self.niter):
            self.r_Axb.append(self.solver.get_residual(i))
        self.x.download(self.sol)
        return self.sol, self.r_Axb, self.niter
    

    def solve(self, M, rhs):
        # self.pyamgx.initialize()
        # self.cfg = self.pyamgx.Config().create_from_file(self.config_file)
        # self.rsc = self.pyamgx.Resources().create_simple(self.cfg)
        # # Create matrices and vectors:
        # self.A = self.pyamgx.Matrix().create(self.rsc)
        # self.b = self.pyamgx.Vector().create(self.rsc)
        # self.x = self.pyamgx.Vector().create(self.rsc)
        # # Create solver:
        # self.solver = self.pyamgx.Solver().create(self.rsc, self.cfg)

        # Upload system:
        self.M = M.astype(np.float64)
        self.rhs = rhs.astype(np.float64)
        self.sol = np.zeros(rhs.shape[0], dtype=np.float64)
        self.A.upload_CSR(self.M)
        self.b.upload(self.rhs)
        self.x.upload(self.sol)

        # Setup and solve system:
        # if should_setup():
        self.solver.setup(self.A)
        self.solver.solve(self.b, self.x)
        self.niter = self.solver.iterations_number


        self.r_Axb = []
        for i in range(self.niter):
            self.r_Axb.append(self.solver.get_residual(i))
        # self.r_final = self.solver.get_residual()
        # self.r0 = self.solver.get_residual(0)

        self.x.download(self.sol)
        
        status = self.solver.status
        # assert status == 'success', f"status:{status}, iterations: {self.niter}. The residual is {self.r_Axb}"
        if status != 'success':
            logging.info(f"status:{status}, iterations: {self.niter}. The residual is {self.r_Axb[-1]}")

        return self.sol, self.r_Axb, self.niter

    def finalize(self):
        # Clean up:
        self.A.destroy()
        self.x.destroy()
        self.b.destroy()
        self.solver.destroy()
        self.rsc.destroy()
        self.pyamgx.finalize()


def test_amgx():
    # usage of AMGX solver
    import sys
    sys.path.append(os.getcwd())
    from engine.common_args import add_common_args
    import argparse
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler()])

    b = np.load("b.npy")

    def get_A0():
        A = sparse.load_npz("A.npz")
        return A
    
    amgxsolver = AmgxSolver(args.amgx_config, get_A0, args.cuda_dir, args.amgx_lib_dir)
    amgxsolver.init()
    x, r_Axb = amgxsolver.run(b)
    amgxsolver.finalize() # remember to finalize the solver, or memory leak will happen


if __name__ == "__main__":
    test_amgx()