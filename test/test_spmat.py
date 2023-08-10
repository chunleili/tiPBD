import taichi as ti
import numpy as np

ti.init(default_fp=ti.f64)
M, N = 20, 20
A_builder = ti.linalg.SparseMatrixBuilder(M, N, max_num_triplets=N * M)


@ti.kernel
def fill_A_kernel(A: ti.sparse_matrix_builder()):
    for i in range(M):
        for j in range(N):
            A[i, j] += 1.0


fill_A_kernel(A_builder)

A = A_builder.build()
b = np.random.rand(N)

# @ti.kernel
# def change_A(A: ti.template()):
#     for i in range(M):
#         for j in range(N):
#             print(A[i, j])

# change_A(A)

print(A[0, 0])

# print(A)
