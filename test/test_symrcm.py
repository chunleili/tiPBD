import taichi as ti
import numpy as np
import scipy
import scipy.sparse as sp
import os, sys
from time import perf_counter
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

sys.path.append(os.getcwd())

from engine.file_utils import export_A_b



A = scipy.sparse.load_npz("A_F0.npz") # load
A0 = A.copy()
b = np.load("b_F0.npy")
print(A.shape)
P=reverse_cuthill_mckee(A,symmetric_mode=True)
A2 = A[P,:][:,P]
print(P.shape)

path = os.getcwd()
print(path)
export_A_b(A2, b=None, dir=path, postfix="symrcm", binary=True)
export_A_b(A, b=None, dir=path, postfix="0", binary=True)

draw = True
if draw:
    fig, axs = plt.subplots(2,figsize=(5, 10))
    axs[0].spy(A0, markersize=5e-3, markevery=2)
    axs[1].spy(A2, markersize=5e-3, markevery=2)

    # plt.tight_layout()
    axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    plt.show()