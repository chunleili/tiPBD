import sys, os
sys.path.append(os.getcwd())

from engine.solver.amg_python import AMG_python, build_levels, setup_smoothers, old_amg_cg_solve
from engine.solver.build_Ps import build_Ps
import numpy as np
import time

import logging
from scipy.sparse import csr_matrix
import scipy
import matplotlib.pyplot as plt

def should_setup():
    return True

def spy(A):
    fig, axs = plt.subplots(1,figsize=(5, 5))
    axs.spy(A, markersize=5e-3, markevery=2)
    # plt.tight_layout()
    axs.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.show()


def fill_A_csr_ti(ist):
    A = scipy.sparse.load_npz("A_0.npz") 
    A = csr_matrix(A)
    print("A_0")
    spy(A)
    return A

def fill_A_csr_ti2(ist):
    A = scipy.sparse.load_npz("A_symrcm.npz") 
    A = csr_matrix(A)
    print("A_symrcm")
    spy(A)
    return A


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    from engine.common_args import add_common_args
    add_common_args(parser)
    args = parser.parse_args()
    args.use_cuda = False
    args.smoother_type = "jacobi"
    print(args)

    from engine.init_extlib import init_extlib

    extlib = init_extlib(args, sim=None)
    args.tol_Axb=1e-8

    class IST:
        pass

    ist = IST()
    b = np.load("b.npy")
    x, r_Axb = AMG_python(b, args, ist, fill_A_csr_ti, should_setup, copy_A=True)
    print(r_Axb)
    print("niter:", len(r_Axb))


    x2, r_Axb2 = AMG_python(b, args, ist, fill_A_csr_ti2, should_setup, copy_A=True)
    print("niter:", len(r_Axb2))


    plt.plot(r_Axb)
    plt.show()
    plt.plot(r_Axb2)
    plt.show()

