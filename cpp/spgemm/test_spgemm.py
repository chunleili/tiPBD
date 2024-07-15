from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, diags, load_npz
import numpy as np
import os
import ctypes
from time import perf_counter

os.chdir(os.path.dirname(__file__))

def prepare_data():
    # small scale data
    # A_offsets = np.array([0, 3, 4, 7, 9], dtype=np.int32)
    # A_columns = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
    # A_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

    # B_offsets = np.array([0, 2, 4, 7, 8], dtype=np.int32)
    # B_columns = np.array([0, 3, 1, 3, 0, 1, 2, 1], dtype=np.int32)
    # B_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    # Ainfo = np.array([4, 4, 9], dtype=np.int32)
    # Binfo = np.array([4, 4, 8], dtype=np.int32)

    # large scale data
    tic = perf_counter()
    A = load_npz('G.npz').tocsr()
    B = A.copy().T.tocsr()
    print("load time:", perf_counter()-tic)

    A_offsets = A.indptr.astype(np.int32)
    A_columns = A.indices.astype(np.int32)
    A_values = A.data.astype(np.float32)

    B_offsets = B.indptr.astype(np.int32)
    B_columns = B.indices.astype(np.int32)
    B_values = B.data.astype(np.float32)

    Ainfo = np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int32)
    Binfo = np.array([B.shape[0], B.shape[1], B.nnz], dtype=np.int32)

    np.savetxt('Ainfo.txt', Ainfo, fmt='%d')
    np.savetxt('Binfo.txt', Binfo, fmt='%d')
    print("info saved:", Ainfo)

    np.savetxt('Aindptr.txt', A_offsets, fmt='%d')
    np.savetxt('Aindices.txt', A_columns, fmt='%d')
    np.savetxt('Adata.txt', A_values, fmt='%f')
    print("A saved")

    np.savetxt('Bindptr.txt', B_offsets, fmt='%d')
    np.savetxt('Bindices.txt', B_columns, fmt='%d')
    np.savetxt('Bdata.txt', B_values, fmt='%f')
    print("B saved")


def run_spgemm():
    from subprocess import call
    # call(["./build/Release/spgemm.exe"], stdout=open("output.txt", "w"))
    call(["./build/Release/spgemm.exe"])
    print("done")


def validify():
    # scipy result
    A = load_npz('G.npz').tocsr().astype(np.float32)
    B = A.copy().T.tocsr().astype(np.float32)
    C_ = A@B

    # cusparse result    
    Cdata = np.loadtxt('C.data.txt')
    Cindices = np.loadtxt('C.indices.txt')
    Cindptr = np.loadtxt('C.indptr.txt')
    C_shape = C_.shape

    # flag = np.allclose(Cdata, C_data_, atol=1e-6)
    # print("validity:", flag)
    C = csr_matrix((Cdata, Cindices, Cindptr), shape=C_shape)
    # print(f"C in scipy: {C_} \n C in cusparse: {C}")
    ...
    diff = C - C_
    diff_data = np.abs(diff.data)
    print("max diff:", diff_data.max())

# run_spgemm()
# validify()

def test_ctypes():
    import numpy.ctypeslib as ctl
    import ctypes
    import numpy as np
    from scipy.sparse import csr_matrix

    libname = 'spgemm.dll'
    libdir = './build/Release'
    lib=ctl.load_library(libname, libdir)

    py_change_spmat = lib.change_spmat
    py_change_spmat.argtypes = [ctl.ndpointer(np.int32, 
                                            flags='aligned, c_contiguous'),  # indptr
                                ctl.ndpointer(np.int32,
                                            flags='aligned, c_contiguous'),  # indices
                                ctl.ndpointer(np.float64,
                                            flags='aligned, c_contiguous'),    # data
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_int]
    # small scale data
    A_offsets = np.array([0, 3, 4, 7, 9], dtype=np.int32)
    A_columns = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
    A_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64)
    A_ = csr_matrix((A_values, A_columns, A_offsets), shape=(4, 4))
    print("A in scipy:")
    print(A_)
    Ainfo = np.array([4, 4, 9], dtype=np.int32) # row, col, nnz
    py_change_spmat(A_offsets, A_columns, A_values, Ainfo[0], Ainfo[1], Ainfo[2])
    A = csr_matrix((A_values, A_columns, A_offsets), shape=(Ainfo[0], Ainfo[1]))
    print("A in c++:")
    print(A)

test_ctypes()