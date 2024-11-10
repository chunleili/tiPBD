import numpy as np
import scipy
import os,sys,shutil, pathlib

import scipy.io

os.chdir(os.getcwd())

hessian1 = scipy.io.mmread("hessian1.mtx")
hessian1_d = hessian1.todense()

hessian_cpp = scipy.io.mmread("hessian_cpp.mtx")
hessian_cpp_d = hessian_cpp.todense()

def csr_is_equal(A, B):
    if A.shape != B.shape:
        print("shape not equal")
        assert False
    diff = A - B
    if diff.nnz == 0:
        print("csr is equal! nnz=0")
        return True
    maxdiff = np.abs(diff.data).max()
    where = np.where(diff.data > 1e-3)
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-3:
        assert False, where
    print("csr is equal!")
    return True

def dense_mat_is_equal(A, B):
    diff = A - B
    maxdiff = np.abs(diff).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-3:
        assert False
    print("is equal!")
    return True


csr_is_equal(hessian1, hessian_cpp)
# dense_mat_is_equal(hessian1, hessian_cpp)
print("done")