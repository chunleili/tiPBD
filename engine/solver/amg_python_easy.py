import numpy as np
from scipy.sparse import csr_matrix
import scipy

def test_amg_python(matA,b):
    import argparse
    import sys,os
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    from engine.common_args import add_common_args
    add_common_args(parser)
    args = parser.parse_args()
    args.use_cuda = False
    args.smoother_type = "jacobi"
    args.tol_Axb=1e-6
    args.maxiter=100
    args.maxiter_Axb=100
    print(args)

    from engine.solver.amg_python import AmgPython

    def get_A0_1():
        A = csr_matrix(matA).astype(np.float32)
        return A
    
    def should_setup():
        return True

    amg = AmgPython(args, get_A0=get_A0_1, should_setup=should_setup)
    x, r_Axb = amg.run(b)

    print(f"AmgPython: {r_Axb[0]:.2e}->{r_Axb[-1]:.2e}")
    print("niter:", len(r_Axb))
    # print("x", x)   
    # assert r_Axb[-1] < args.tol_Axb * r_Axb[0]
    return x, r_Axb


    


if __name__ == "__main__":
    A = scipy.sparse.load_npz("result/test_A/A/A_F1.npz") 
    b = np.load("result/test_A/A/b_F1.npy")
    print("read done")
    x,r_Axb=test_amg_python(A,b)
    import matplotlib.pyplot as plt
    plt.plot(r_Axb)
    plt.yscale('log')
    plt.show()