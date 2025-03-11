def amg_cuda_easy(matA, b, tol=1e-6,maxiter=100, build_P_method="UA", smoother_type="jacobi"):
    """Easy to use version of AMG-CUDA for a given A and b.
    Performance may be not optimal, but it is useful for testing and debugging.

    Except the given args, the args defined in common_args also works!
    """
    from scipy.sparse import csr_matrix
    import argparse
    import sys,os
    import numpy as np
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    from engine.common_args import add_common_args
    add_common_args(parser)
    args = parser.parse_args()
    args.smoother_type = smoother_type
    args.tol_Axb=tol
    args.maxiter_Axb=maxiter
    args.build_P_method = build_P_method

    from engine.init_extlib import init_extlib
    extlib = init_extlib(args,"")

    from engine.solver.amg_cuda import AmgCuda

    def get_A0():
        A = csr_matrix(matA).astype(np.float32)
        return A
    
    def should_setup():
        return True
    
    def AMG_A():
        A = get_A0()
        extlib.fastmg_set_A0(A.data, A.indices, A.indptr, A.shape[0], A.shape[1], A.nnz)

    if build_P_method == "PCG":
        amg = AmgCuda(args, extlib, get_A0=get_A0, fill_A_in_cuda=AMG_A, should_setup=should_setup, only_PCG=True)
    else:
        amg = AmgCuda(args, extlib, get_A0=get_A0, fill_A_in_cuda=AMG_A, should_setup=should_setup)
    x, r_Axb = amg.run(b)
    print(f"AmgCuda: {r_Axb[0]:.2e}->{r_Axb[-1]:.2e}")
    # print(r_Axb)
    # print("x", x)   
    print("niter:", len(r_Axb))
    return x, r_Axb


if __name__ == "__main__":
    import scipy
    import numpy as np
    A = scipy.sparse.load_npz("result/test_A/A/A_F1.npz") 
    b = np.load("result/test_A/A/b_F1.npy")
    x,r_Axb=amg_cuda_easy(A,b)
    import matplotlib.pyplot as plt
    plt.plot(r_Axb)
    plt.yscale('log')
    plt.show()