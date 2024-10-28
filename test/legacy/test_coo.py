import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import numpy as np
import taichi as ti

ti.init()

n = 8
m = 3
lam = np.random.rand(m, 1)

gradC_vec = np.random.rand(m, 4, 3)
print("gradC_vec")
print(gradC_vec)

tet_indices = np.random.randint(low=0, high=n, size=(m, 4))

# fill gradC
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

gradC_coo = coo_matrix((val, (row, col)), shape=(m, n * 3))
print("gradC_coo\n", gradC_coo)
print("gradC_coo.todense()\n", gradC_coo.todense())

# ---------------------------------------------------------------------------- #
#                             taichi sparse matrix                             #
# ---------------------------------------------------------------------------- #

gradC_builder = ti.linalg.SparseMatrixBuilder(m, 3 * n, max_num_triplets=12 * m)


@ti.kernel
def fill(
    A: ti.types.sparse_matrix_builder(),
    gradC_vec: ti.types.ndarray(),
    tet_indices: ti.types.ndarray(dtype=ti.math.vec4),
):
    # fill gradC
    for j in range(m):
        ind = tet_indices[j]
        for p in range(4):
            for d in range(3):
                pid = ind[p]
                A[j, 3 * pid + d] += gradC_vec[j, p, d]


fill(gradC_builder, gradC_vec, tet_indices)

gradC_mat = gradC_builder.build()
print(gradC_mat)


# ---------------------------------------------------------------------------- #
# scipy
# schur = gradC_coo @ gradC_coo.T
# taichi
# schur_ti = gradC_mat @ gradC_mat.transpose()
# print("schur_ti\n", schur_ti)

# scipy
inv_mass = np.random.rand(n).repeat(3)
inv_mass_coo = scipy.sparse.diags(inv_mass)
schur = gradC_coo @ inv_mass_coo @ gradC_coo.T
print("schur\n", schur.todense())

# taichi
inv_mass_builder = ti.linalg.SparseMatrixBuilder(3 * n, 3 * n, max_num_triplets=3 * n)


@ti.kernel
def fill_mass(A: ti.types.sparse_matrix_builder(), inv_mass: ti.types.ndarray()):
    for i in range(3 * n):
        A[i, i] += inv_mass[i]


fill_mass(inv_mass_builder, inv_mass)
inv_mass_mat = inv_mass_builder.build()
schur_ti = gradC_mat @ inv_mass_mat @ gradC_mat.transpose()
print("schur_ti\n", schur_ti)