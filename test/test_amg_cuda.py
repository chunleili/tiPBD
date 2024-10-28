import numpy as np
from scipy.sparse import csr_matrix
import scipy

def test_amg_cuda():
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

    from engine.init_extlib import init_extlib
    extlib = init_extlib(args,"")

    from engine.solver.amg_cuda import AmgCuda

    b = np.load("b.npy")

    def get_A0():
        A = scipy.sparse.load_npz("A.npz") 
        A = csr_matrix(A)
        return A
    
    def should_setup():
        return True
    
    def AMG_A():
        A = get_A0()
        extlib.fastmg_set_A0(A.data, A.indices, A.indptr, A.shape[0], A.shape[1], A.nnz)


    amg = AmgCuda(args, extlib, get_A0=get_A0, AMG_A=AMG_A, should_setup=should_setup)
    x, r_Axb = amg.run(b)
    print(r_Axb)
    print("x", x)   
    print("niter:", len(r_Axb))
    assert r_Axb[-1] < args.tol_Axb * r_Axb[0]


if __name__ == "__main__":
    test_amg_cuda()