import scipy
import numpy as np
import taichi as ti


def test_matrix_free_A():
    gradC = np.load("gradC.npy")
    inv_mass = np.loadtxt("inv_mass.txt")
    alpha_tilde = np.loadtxt("alpha_tilde.txt")
    tet_indices = np.loadtxt("tet_indices.txt", dtype=np.int32)
    A_true = scipy.io.mmread("A_true.mtx")
    gradC_mat = scipy.io.mmread("gradC_mat.mtx")
    inv_mass_mat = scipy.io.mmread("inv_mass_mat.mtx")

    # gradC = np.ones((2, 4, 3))
    # inv_mass = np.ones((5))
    # inv_mass[0] = 2
    # alpha_tilde = np.ones((2))
    # tet_indices = np.array([[0,1,2,3],[1,2,3,4]], dtype=np.int32)

    M = gradC.shape[0]
    N = inv_mass.shape[0]

    # true A based on mat product
    gradC_mat = assemble_gradC_mat(gradC, tet_indices, N)
    inv_mass_mat = assemble_inv_mass_mat(inv_mass)
    gradC_invmass_true = gradC_mat @ inv_mass_mat
    gradC_invmass_gradCT_true = gradC_invmass_true @ gradC_mat.T

    gradC_12 = gradC.reshape((M, 12))
    inv_mass3 = np.repeat(inv_mass, 3)
    col_indices = np.zeros((M, 12), dtype=np.int32)
    for i in range(M):
        for j in range(4):
            for k in range(3):
                col_indices[i, j * 3 + k] = tet_indices[i, j] * 3 + k
    gradC_M_ = spmat12_diamat_prod(gradC_12, inv_mass3, col_indices)

    gradC_M_gradCT = spmat12_spmat12T_prod(gradC_M_, gradC_12, col_indices)

    gradC_invmass = gradC.copy()
    for i in range(M):
        ia, ib, ic, id = tet_indices[i]
        gradC_invmass[i, 0, :] = gradC[i, 0, :] * inv_mass[ia]
        gradC_invmass[i, 1, :] = gradC[i, 1, :] * inv_mass[ib]
        gradC_invmass[i, 2, :] = gradC[i, 2, :] * inv_mass[ic]
        gradC_invmass[i, 3, :] = gradC[i, 3, :] * inv_mass[id]

    gradC_invmass_gradCT = gradC_invmass.copy()
    for i in range(M):
        for j in range(M):
            for ni, ii in enumerate(tet_indices[i]):
                i_idx = i * 4 + ni
                for nj, jj in enumerate(tet_indices[j]):
                    j_idx = j * 4 + nj
                    if ii == jj:
                        gradC_invmass_gradCT[i, ni, :] += gradC_invmass[i, ni, :] * gradC_invmass[j, nj, :]

    gradC_invmass_gradCT = gradC_invmass_gradCT.reshape((M, 12))
    gradC_invmass_gradCT_plus_alpha = np.zeros((M, 13), dtype=np.float32)
    for i in range(M):
        gradC_invmass_gradCT_plus_alpha[i, 0:12] = gradC_invmass_gradCT[i, 0:12]

        ind = tet_indices[i]
        flag = False
        for p in range(4):
            for dim in range(3):
                col_ind = ind[p] * 3 + dim
                col_num = p * 3 + dim
                if col_ind == i:
                    gradC_invmass_gradCT_plus_alpha[i, col_num] += alpha_tilde[i]
                    flag = True
        if not flag:
            gradC_invmass_gradCT_plus_alpha[i, 12] = alpha_tilde[i]
    ...


def assemble_gradC_mat(gradC, tet_indices, N):
    M = gradC.shape[0]
    A = np.zeros((M, N * 3), dtype=np.float32)
    for j in range(tet_indices.shape[0]):
        ind = tet_indices[j]
        for p in range(4):
            for d in range(3):
                pid = ind[p]
                A[j, 3 * pid + d] += gradC[j, p, d]
    return A


def assemble_inv_mass_mat(inv_mass):
    N = inv_mass.shape[0]
    A = np.zeros((3 * N, 3 * N), dtype=np.float32)
    for i in range(N):
        A[3 * i, 3 * i] = inv_mass[i]
        A[3 * i + 1, 3 * i + 1] = inv_mass[i]
        A[3 * i + 2, 3 * i + 2] = inv_mass[i]
    return A


def matmatprod(A, B):
    M = A.shape[0]
    N = B.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            for k in range(N):
                C[i, j] = A[i, k] @ B[k, j]
    return C


def spmat12_diamat_prod(A, D, col_ind):
    """
    A@D
    A(Mx12): sparse matrix, csr format, each line has 12 nonzeros
    D(M): diagonal matrix, dense format
    col_ind(Mx12): column indices of A
    """
    M = A.shape[0]
    C = np.zeros_like(A)
    for i in range(M):
        for d in range(12):
            index = col_ind[i, d]
            C[i, d] = A[i, d] * D[index]
    return C


def spmat_diamat_prod(A, D, col_ind, ind_ptr):
    """
    A@D
    A(Mx12): sparse matrix, csr format, each line has 12 nonzeros
    D(M): diagonal matrix, dense format
    col_ind(Mx12): column indices of A
    ind_ptr(M+1): index pointer of A
    """
    M = A.shape[0]
    for i in range(M):
        for d in range(12):
            index = col_ind[i, d]
            C[i, d] = A[i, d] * D[index]
    return C


def spmat12_spmat12T_prod(A, B, col_ind):
    """
    A@B
    A(Mx12): sparse matrix, csr format, each line has 12 nonzeros
    B(Mx12): sparse matrix, csr format, each line has 12 nonzeros
    col_ind(Mx12): column indices of A and B
    """
    M = A.shape[0]
    C = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            for ki in range(12):
                for kj in range(12):
                    ni = col_ind[i, ki]
                    nj = col_ind[j, kj]
                    if ni == nj:
                        C[i, j] += A[i, ki] * B[j, kj]
    return C


def spmat12_add_diag(A, col_ind, diag):
    """
    A + alpha * I
    A(Mx12): sparse matrix, csr format, each line has 12 nonzeros
    diag(M): diagonal matrix, dense format
    """
    M = A.shape[0]
    C = np.zeors((M, M), dtype=np.float32)
    for i in range(M):
        C[i, i] = A[i, i] + diag[i]
    return C


if __name__ == "__main__":
    test_matrix_free_A()
