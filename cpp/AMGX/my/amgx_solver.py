import numpy as np
import scipy.sparse as sparse
import os

class AMGXSolver:
    def __init__(self, config_file):
        self.config_file = config_file
        amgx_lib_dir = "D:/Dev/AMGX/build/Release"
        cuda_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
        os.add_dll_directory(amgx_lib_dir)
        os.add_dll_directory(cuda_dir)
        import pyamgx

        self.pyamgx = pyamgx
        pyamgx.initialize()
        cfg = pyamgx.Config().create_from_file(self.config_file)
        self.rsc = pyamgx.Resources().create_simple(cfg)
        # Create matrices and vectors:
        self.A = pyamgx.Matrix().create(self.rsc)
        self.b = pyamgx.Vector().create(self.rsc)
        self.x = pyamgx.Vector().create(self.rsc)
        # Create solver:
        self.solver = pyamgx.Solver().create(self.rsc, cfg)

    def solve(self, M, rhs):
        # Upload system:
        M = M.astype(np.float64)
        rhs = rhs.astype(np.float64)
        sol = np.zeros(rhs.shape[0], dtype=np.float64)
        self.A.upload_CSR(M)
        self.b.upload(rhs)
        self.x.upload(sol)

        # Setup and solve system:
        self.solver.setup(self.A)
        self.solver.solve(self.b, self.x)

        # Download solution
        self.x.download(sol)
        print("pyamgx solution: ", sol)


    def finalize(self):
        # Clean up:
        self.A.destroy()
        self.x.destroy()
        self.b.destroy()
        self.solver.destroy()
        self.rsc.destroy()
        self.pyamgx.finalize()


if __name__ == "__main__":
    M = sparse.load_npz("result/latest/A/A_20-0.npz")
    rhs = np.load("result/latest/A/b_20-0.npy")
    solver = AMGXSolver("D:/Dev/AMGX/src/configs/agg_cheb4.json")
    solver.solve(M, rhs)
    solver.finalize()