# https://stackoverflow.com/a/42831307/19253199

import numpy.ctypeslib as ctl
import ctypes
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

libname = 'testLib.dll'
libdir = './build/Release'
lib=ctl.load_library(libname, libdir)

py_add_one = lib.add_one
py_add_one.argtypes = [ctypes.c_int]
value = 5
results = py_add_one(value)
print(results)



py_print_array = lib.print_array
py_print_array.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'), 
                           ctypes.c_int]
A = np.array([1.4,2.6,3.0], dtype=np.float64)
py_print_array(A, 3)


py_change_array = lib.change_array
py_change_array.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'), 
                           ctypes.c_int]
A = np.array([1.4,2.6,3.0], dtype=np.float64)
py_change_array(A, A.shape[0])
print(A)


from scipy.sparse import csr_matrix
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
print(A_.toarray())
Ainfo = np.array([4, 4, 9], dtype=np.int32) # row, col, nnz
py_change_spmat(A_offsets, A_columns, A_values, Ainfo[0], Ainfo[1], Ainfo[2])
A = csr_matrix((A_values, A_columns, A_offsets), shape=(Ainfo[0], Ainfo[1]))
print(A.toarray())