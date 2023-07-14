import scipy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmwrite
import numpy as np
import taichi as ti

ti.init()

# prepare data
n = 8
m = 3
lam = np.random.rand(m, 1)
gradC_vec = np.random.rand(m, 4, 3)
tet_indices = np.random.randint(low=0, high=n, size=(m, 4))
inv_mass = np.random.rand(n).repeat(3)


# ---------------------------------------------------------------------------- #
#                                     scipy                                    #
# ---------------------------------------------------------------------------- #
# fill gradC
def fill_gradC_scipy(gradC_vec):
    row = np.zeros(12 * m, dtype=np.int32)
    col = np.zeros(12 * m, dtype=np.int32)
    val = np.zeros(12 * m, dtype=np.float32)
    for j in range(m):
        # ia,ib,ic,id = np.random.randint(low=0, high=n,size=4)
        ia, ib, ic, id = tet_indices[j]
        val[12 * j + 0 * 3 : 12 * j + 0 * 3 + 3] = gradC_vec[j, 0]
        val[12 * j + 1 * 3 : 12 * j + 1 * 3 + 3] = gradC_vec[j, 1]
        val[12 * j + 2 * 3 : 12 * j + 2 * 3 + 3] = gradC_vec[j, 2]
        val[12 * j + 3 * 3 : 12 * j + 3 * 3 + 3] = gradC_vec[j, 3]
        row[12 * j : 12 * j + 12] = j
        col[12 * j + 3 * 0 : 12 * j + 3 * 0 + 3] = 3 * ia, 3 * ia + 1, 3 * ia + 2
        col[12 * j + 3 * 1 : 12 * j + 3 * 1 + 3] = 3 * ib, 3 * ib + 1, 3 * ib + 2
        col[12 * j + 3 * 2 : 12 * j + 3 * 2 + 3] = 3 * ic, 3 * ic + 1, 3 * ic + 2
        col[12 * j + 3 * 3 : 12 * j + 3 * 3 + 3] = 3 * id, 3 * id + 1, 3 * id + 2

    res = csr_matrix((val, (row, col)), shape=(m, n * 3))
    # print("gradC_scipy\n", res)
    # print("gradC_scipy.todense()\n", res.todense())
    return res


gradC_scipy = fill_gradC_scipy(gradC_vec)

# Assemble A
inv_mass_mat = scipy.sparse.diags(inv_mass)
A = gradC_scipy @ inv_mass_mat @ gradC_scipy.T
print("A\n", A.todense())


# ---------------------------------------------------------------------------- #
#                                  matrix free                                 #
# ---------------------------------------------------------------------------- #
def matmul(A, B):
    # A: m x n
    # B: n x k
    # C: m x k
    m, n = A.shape
    n, k = B.shape
    C = np.zeros((m, k), dtype=np.float32)
    for i in range(m):
        for j in range(k):
            for l in range(n):
                C[i, j] += A[i, l] * B[l, j]
