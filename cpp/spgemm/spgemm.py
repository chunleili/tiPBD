''' calculation of A@A.T using cusparse'''
import scipy
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, diags, load_npz
from scipy.sparse import csr_matrix
import numpy as np
import os
import ctypes
from time import perf_counter
import numpy.ctypeslib as ctl
import ctypes
import numpy as np

os.chdir(os.path.dirname(__file__))

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
lib = ctypes.cdll.LoadLibrary('./build/Release/spgemm.dll')

def test_spgemm():
    # generate data
    print("generating data...")
    A = scipy.sparse.random(1000, 1000, density=0.01, format='csr', dtype=np.float32)
    print("generation done.")

    # parameters types defining
    lib.spgemm.argtypes = [
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # Aindptr
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # Aindices
        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),  # Adata
        ctypes.c_int,  # A_nrows
        ctypes.c_int,  # A_ncols
        ctypes.c_int,  # A_nnz
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # Bindptr
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # Bindices
        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),  # Bdata
        ctypes.c_int,  # B_nrows
        ctypes.c_int,  # B_ncols
        ctypes.c_int,  # B_nnz
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # Cindptr
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # Cindices
        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),  # Cdata
        ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),  # C_nnz
    ]
    

    print("C in scipy:")
    tic = perf_counter()
    C_ = A @ A.T
    toc = perf_counter()
    print("time:", toc-tic)
    print(C_)

    print("C in cuda:")
    B = A.T.copy().tocsr().astype(np.float32)
    C_nnz = np.array([A.shape[0]*100], dtype=np.int32) 
    # FIXME: Now we have to guess the nnz of C before passing it to cuda. Small nnz will cause error like this: OSError: exception: access violation writing 0x000003BE000003B1
    C_indptr = np.zeros(A.shape[0]+1, dtype=np.int32)
    C_indices = np.zeros(C_nnz[0], dtype=np.int32)
    C_data = np.zeros(C_nnz[0], dtype=np.float32)
    tic = perf_counter()
    lib.spgemm(A.indptr, A.indices, A.data, A.shape[0], A.shape[1], A.nnz, 
               B.indptr, B.indices, B.data, B.shape[0], B.shape[1], B.nnz, 
               C_indptr, C_indices, C_data, C_nnz)
    toc = perf_counter()
    print("time:", toc-tic)
    C = csr_matrix((C_data, C_indices, C_indptr), shape=(A.shape[0], B.shape[1]))
    print(C)

    # validate
    diff = C - C_
    diff_data = np.abs(diff.data)
    print("max diff:", diff_data.max())

test_spgemm()
