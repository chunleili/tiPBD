import numpy as np
from scipy.sparse import csr_matrix
import scipy
import taichi as ti


def test_solvers():
    import argparse
    import sys,os
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    from engine.common_args import add_common_args
    add_common_args(parser)
    args = parser.parse_args()
    args.smoother_type = "jacobi"
    args.tol_Axb=1e-6
    args.maxiter=100
    args.maxiter_Axb=100
    print(args)

    ti.init()

    from engine.init_extlib import init_extlib
    extlib = init_extlib(args,"")

    from engine.solver.amg_cuda import AmgCuda
    from engine.solver.iterative_solver import GaussSeidelSolver

    b = np.load("test/data/b.npy")

    def get_A0():
        A = scipy.sparse.load_npz("test/data/A.npz") 
        A = csr_matrix(A)
        return A
    
    def should_setup():
        return True
    
    def AMG_A():
        A = get_A0()
        extlib.fastmg_set_A0(A.data, A.indices, A.indptr, A.shape[0], A.shape[1], A.nnz)

    args.tol_Axb = 1e-9
    args.maxiter_Axb = 100

    amg = AmgCuda(args, extlib, get_A0=get_A0, fill_A_in_cuda=AMG_A, should_setup=should_setup)
    x1, r_Axb1 = amg.run(b)
    for i in range(len(r_Axb1)):
        print(f"{r_Axb1[i]}")

    gs = GaussSeidelSolver(args=args, get_A0=get_A0, calc_residual_every_iter=True)
    x2, r_Axb2 = gs.run(b)
    for i in range(len(r_Axb2)):
        print(f"{r_Axb2[i]}")
    
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(r_Axb1, label="AMG")
        plt.plot(r_Axb2, label="GS")
        plt.legend()
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    test_solvers()