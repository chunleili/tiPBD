'''
sparse matrix times vector for taichi
sparse matrix is given by csr format

csr: the column indices for row i are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in data[indptr[i]:indptr[i+1]].

see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array

A.indptr #first element is 0, last element is nnz   
A.indices
A.data
'''

import numpy as np
import scipy.sparse as sp

dat = np.array([1,2,3,4,5,6], dtype=np.float32)
ind = np.array([0,2,0,1,2,1], dtype=np.int32)
ptr = np.array([0,2,5], dtype=np.int32)
v = np.array([1,2,3], dtype=np.float32)
# N=6
# dat = np.random.rand(N).astype(np.float32)
# ind = np.random.randint(0, N,   6).astype(np.int32)
# ptr = np.random.randint(0, N, 3).astype(np.int32)
# ptr[0] = 0
# v = np.random.rand(3).astype(np.float32)    

def spmv_csr(dat, ind, ptr, v):
    res = np.zeros((len(ptr)-1), dtype=np.float32)
    dv = np.ones_like(ind, dtype=np.float32) #dat * v    
    v_ = np.zeros_like(ind, dtype=np.float32) #v[ind]

    for i in range(len(ind)):
        idx = ind[i]
        v_[i] = v[idx]
        dv[i] = v[idx] * dat[i]
    for k in range(len(v)-1):
        for j in range(ptr[k], ptr[k+1]):
            res[k] += dv[j]
    return res

def spmv_scipy():
    A = sp.csr_array((dat, ind, ptr))   
    res = A@v
    return res


print("result of scipy: ", spmv_scipy())

print("result of my: ", spmv_csr(dat, ind, ptr, v))  