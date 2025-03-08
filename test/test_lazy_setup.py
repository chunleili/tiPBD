import numpy as np
from scipy.sparse import csr_matrix
import scipy

def test_amg_cuda(label="F1",use_outer_Ps=False):
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

    dir = "result/case180-0308-bunny-interval300/A/"
    b = np.load(dir+f"b_{label}.npy")

    def get_A0():
        A = scipy.sparse.load_npz(dir+f"A_{label}.npz") 
        A = csr_matrix(A)
        return A
    
    def should_setup():
        return True
    
    def AMG_A():
        A = get_A0()
        extlib.fastmg_set_A0(A.data, A.indices, A.indptr, A.shape[0], A.shape[1], A.nnz)

    if use_outer_Ps:
        Ps = scipy.sparse.load_npz(dir+f"/P_{label}.npz")
        Ps = [Ps]
    else :
        Ps = None

    amg = AmgCuda(args, extlib, get_A0=get_A0, fill_A_in_cuda=AMG_A, should_setup=should_setup, outer_Ps=Ps)
    x, r_Axb = amg.run(b)

    # Ps = amg.Ps
    # for i,P in enumerate(Ps):
    #     scipy.sparse.save_npz(dir+f"/P_{label}L{i}.npz", P)

    print(r_Axb)
    print("x", x)   
    print("niter:", len(r_Axb))
    # assert r_Axb[-1] < args.tol_Axb * r_Axb[0]
    
    # np.savetxt(dir+"residual_"+label+int(use_outer_Ps)+".txt", r_Axb)
    np.savetxt(dir+f"residual_{label}-{use_outer_Ps}.txt", r_Axb)

    if use_outer_Ps:
        ax.plot(r_Axb, label=f"{label}-lazy", linestyle=":", marker="o", linewidth=2)
    else:
        ax.plot(r_Axb, label=f"{label}-full")
    # ax.plot(r_Axb, label=f"{label}-{use_outer_Ps}")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylabel("Dual Residual")
    ax.set_xlabel("Iteration")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    test_amg_cuda("F1",use_outer_Ps=True)
    test_amg_cuda("F10",use_outer_Ps=True)
    test_amg_cuda("F50",use_outer_Ps=True)
    test_amg_cuda("F100",use_outer_Ps=True)
    test_amg_cuda("F1",use_outer_Ps=False)
    test_amg_cuda("F10",use_outer_Ps=False)
    test_amg_cuda("F50",use_outer_Ps=False)
    test_amg_cuda("F100",use_outer_Ps=False)
    
    # save the  matplotlib object
    import pickle
    pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

    plt.show()
    