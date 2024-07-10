import sys
import os
from pathlib import Path
from scipy.sparse import csr_matrix, diags
sys.path.append(os.getcwd())
thisdir = Path(__file__).resolve().parent
print(thisdir/'build/Release')
sys.path.append(str(thisdir/'build/Release'))  # 添加模块搜索路径
sys.path.append(str(thisdir/'build/Debug'))  # 添加模块搜索路径

import fillA
import numpy as np
import copy

def test_fill_csr():
    indptr = np.array([0, 3, 4, 7, 9], dtype=np.int32)
    col = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

    A = csr_matrix((val, col, indptr), shape=(4, 4))
    nrows, ncols = A.shape
    nnz = A.nnz
    print("A in python:")
    print(A.toarray())

    print("A in C++:")
    fillA.fill_csr(nrows, ncols, nnz, indptr, col, val)

    pos = np.ones((10,3), dtype=np.float32) * 1.5
    fillA.pass_vec3f(pos)

    edge = np.ones((10,2), dtype=np.int32)
    fillA.pass_vec2i(edge)

def test_compute_C_and_gradC():
    pos = np.ones((10,3), dtype=np.float32) * 1.5
    edge = np.ones((10,2), dtype=np.int32)

    constraints = np.ones((10), dtype=np.float32)
    rest_len = np.ones((10), dtype=np.float32)
    gradC = np.zeros((10, 2, 3), dtype=np.float32)
    fillA.compute_C_and_gradC(edge, pos, constraints, rest_len, gradC)


def test_spgemm():
    indptr = np.array([0, 3, 4, 7, 9], dtype=np.int32)
    col = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    A = csr_matrix((val, col, indptr), shape=(4, 4))
    nrows, ncols = A.shape
    nnz = A.nnz

    indptr2 = np.array([0, 3, 4, 7, 9], dtype=np.int32)
    col2 = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
    val2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    B = csr_matrix((val, col, indptr), shape=(4, 4))
    nrows2, ncols2 = B.shape
    nnz2 = B.nnz

    print(A.toarray())
    print(B.toarray())

    C_ = A@B
    print(C_.toarray())

    C = csr_matrix((nrows, ncols), dtype=np.float32)
    fillA.spgemm(nrows, ncols, nnz, indptr, col, val,
                    nrows2, ncols2, nnz2, indptr2, col2, val2,
                    C.indptr, C.indices, C.data)
    

def test_spGMGT_plus_alpha():
    indptr = np.array([0, 3, 4, 7, 9], dtype=np.int32)
    col = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    A = csr_matrix((val, col, indptr), shape=(4, 4))
    nrows, ncols = A.shape
    nnz = A.nnz

    M = diags([1, 2, 3, 4], format='csr')
    nrows2, ncols2 = M.shape
    nnz2 = M.nnz
    val2 = M.data
    indptr2 = M.indptr
    col2 = M.indices

    print("python:")
    C_ = A@M@A.T
    # diag + 0.01
    alpha_diag = diags([0.01, 0.01, 0.01, 0.01], format='csr')
    C_ = C_ + alpha_diag
    print(C_.toarray())

    print("C++:")
    C = csr_matrix((nrows, nrows), dtype=np.float32)

    fillA.spGMGT_plus_alpha(nrows, ncols, nnz, indptr, col, val,
                nrows2, ncols2, nnz2, indptr2, col2, val2,
                C.data, C.indices, C.indptr, 0.01)
    print("back in python:")
    print(C.toarray())


def test_pass_by_ref():
    a = np.ones((10), dtype=np.float32)
    fillA.pass_by_ref(a)
    print(a)
    
def test_pass_eigen_by_ref():
    a = np.ones((10), dtype=np.float32)
    fillA.pass_eigen_by_ref(a)
    print(a)


test_pass_by_ref()