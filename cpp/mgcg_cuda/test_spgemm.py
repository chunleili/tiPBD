from scipy.io import mmread, mmwrite
import numpy as np
import os
import ctypes

# mat = mmread('G.mtx').tocsr()


# data_contig = np.ascontiguousarray(mat.data, dtype=np.float32)
# indices_contig = np.ascontiguousarray(mat.indices, dtype=np.int32)
# indptr_contig = np.ascontiguousarray(mat.indptr, dtype=np.int32)
# cuda_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin"
# os.add_dll_directory(cuda_dir)
# g_vcycle = ctypes.cdll.LoadLibrary('./cpp/mgcg_cuda/lib/fast-vcycle-gpu.dll')
# g_vcycle.fastmg_set_lv_csrmat(data_contig.ctypes.data, data_contig.shape[0],
#                                 indices_contig.ctypes.data, indices_contig.shape[0],
#                                 indptr_contig.ctypes.data, indptr_contig.shape[0],
#                                 mat.shape[0], mat.shape[1], mat.nnz)


A_offsets = np.array([0, 3, 4, 7, 9], dtype=np.int32)
A_columns = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
A_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

B_offsets = np.array([0, 2, 4, 7, 8], dtype=np.int32)
B_columns = np.array([0, 3, 1, 3, 0, 1, 2, 1], dtype=np.int32)
B_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

os.chdir('cpp/mgcg_cuda')

np.savetxt('Aindptr.txt', A_offsets, fmt='%d')
np.savetxt('Aindices.txt', A_columns, fmt='%d')
np.savetxt('Adata.txt', A_values, fmt='%f')

np.savetxt('Bindptr.txt', B_offsets, fmt='%d')
np.savetxt('Bindices.txt', B_columns, fmt='%d')
np.savetxt('Bdata.txt', B_values, fmt='%f')

Ainfo = np.array([4, 4, 9], dtype=np.int32)
Binfo = np.array([4, 4, 8], dtype=np.int32)

np.savetxt('Ainfo.txt', Ainfo, fmt='%d')
np.savetxt('Binfo.txt', Binfo, fmt='%d')




# G = mmread('G.mtx').tocsr()
# print(G)

# os.chdir('cpp/mgcg_cuda')

# Grows, Gcols = G.shape
# Gnnz = G.nnz
# Ginfo = np.array([Grows, Gcols, Gnnz], dtype=np.int32)

# np.savetxt('Aindptr.txt', G.indptr, fmt='%d')
# np.savetxt('Aindices.txt', G.indices, fmt='%d')
# np.savetxt('Adata.txt', G.data, fmt='%f')
# np.savetxt('Ainfo.txt', Ginfo, fmt='%d')


# GT = G.transpose()
# GTrows, GTcols = GT.shape
# GTnnz = GT.nnz
# GTinfo = np.array([GTrows, GTcols, GTnnz], dtype=np.int32)

# np.savetxt('Bindptr.txt', GT.indptr, fmt='%d')
# np.savetxt('Bindices.txt', GT.indices, fmt='%d')
# np.savetxt('Bdata.txt', GT.data, fmt='%f')
# np.savetxt('Binfo.txt', GTinfo, fmt='%d')
