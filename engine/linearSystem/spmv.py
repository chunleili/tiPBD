'''
sparse matrix times vector for taichi
sparse matrix is given by csr format

csr: the column indices for row i are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in data[indptr[i]:indptr[i+1]].

see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array

A.indptr #first element is 0, last element is nnz   
A.indices 
A.data

sildes: 【金山文档】 2023-10-25 关于spmv CSR和Matrix-free
https://kdocs.cn/l/clW7ztNAdwaz
'''

import numpy as np
import scipy.sparse as sp



def spmv_csr(dat, ind, ptr, v):
    res = np.zeros((len(ptr)-1), dtype=np.float64)
    dv = np.ones_like(ind, dtype=np.float64) #dat * v    
    v_ = np.zeros_like(ind, dtype=np.float64) #v[ind]

    for i in range(len(ind)):
        idx = ind[i]
        v_[i] = v[idx]
        dv[i] = v[idx] * dat[i]
    for k in range(len(ptr)-1):
        for j in range(ptr[k], ptr[k+1]):
            res[k] += dv[j]
    return res


def test_small_case():
    # small case for tutorial
    dat = np.array([1,2,3,4,5,6], dtype=np.float32)
    ind = np.array([0,2,0,1,2,1], dtype=np.int32)
    ptr = np.array([0,2,5], dtype=np.int32)
    v = np.array([1,2,3], dtype=np.float32)

    res0 = spmv_csr(dat, ind, ptr, v)
    print("res0: ", res0)

    A = sp.csr_array((dat, ind, ptr))   
    res1 = A@v
    print("res1: ", res1)

    diff = res0 - res1
    print("diff: ", diff)


def test_large_case(N=1000):
    # large case for test
    # print("generating large case")
    A = sp.random(N, N, density=0.01, format='csr')
    dat = A.data
    ind = A.indices
    ptr = A.indptr
    v = np.random.rand(N).astype(np.float64)
    # print("generated large case")

    import time
    t0 = time.time()
    res0 = A@v
    t1 = time.time()
    res1 = spmv_csr(dat, ind, ptr, v)
    t2 = time.time()

    # print("first 10 elements of scipy: ", res0[:10])
    # print("first 10 elements of spmv: ", res1[:10])

    # print(f"\ntime of scipy:  {t1-t0:.2g}s")
    # print(f"time of my: {t2-t1:.2g}s\n")

    # diff = res0 - res1
    # print("max diff: ", np.max(diff))
    # print("argmax diff: ", np.argmax(diff))
    # print("diff[argmax]: ", diff[np.argmax(diff)])
    # print("res0[argmax]: ", res0[np.argmax(diff)])
    # print("res1[argmax]: ", res1[np.argmax(diff)])

    # print(f"res0.shape: {res0.shape}")
    # print(f"res1.shape: {res1.shape}")
    # print(f"res0.dtype: {res0.dtype}")
    # print(f"res1.dtype: {res1.dtype}")

    # indptr = A.indptr
    
    # print(f"last of indptr is {indptr[-1]}")

    if np.allclose(res0, res1, atol=1e-6, rtol=1e-3):
        print("result is the same")
        return True
    else:       
        print("result is different")
        res0 = A@v
        res1 = spmv_csr(dat, ind, ptr, v)
        return False
    


if __name__ == "__main__":
    # test_small_case()
    for _ in range(30):
        N = np.random.randint(1000, 10000)
        print(f"case:{_}\tN: {N}")
        suc = test_large_case(N)
        if not suc:
            exit()
            break