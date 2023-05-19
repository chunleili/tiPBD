import numpy as np
import taichi as ti
import os, sys

sys.path.append(os.getcwd())

n = 4
m = 1
r = np.zeros(3 * n)
gradC = np.zeros((m, 3 * n))
lam = np.zeros((m, 1))
C = np.zeros((m, 1))
dx = np.zeros((3 * n, 1))
alpha_tilde_inv = np.eye(m)
M = np.eye(3 * n, 3 * n)


# fill in the values of the matrices and vectors
def fill_matrices(mat):
    n = mat.shape[0]
    m = mat.shape[1]
    for i in range(n):
        for j in range(m):
            mat[i, j] = np.random.rand()


def fill_vectors(vec):
    n = vec.shape[0]
    for i in range(n):
        vec[i] = np.random.rand()


def fill_eye(mat):
    n = mat.shape[0]
    for i in range(n):
        mat[i, i] = np.random.rand()


fill_eye(M)
fill_matrices(gradC)
fill_vectors(lam)
fill_vectors(C)
fill_vectors(dx)
fill_eye(alpha_tilde_inv)


# matrix multiplication using numpy for true value
r_true = np.dot(M, dx) + np.dot(gradC.T, lam) + np.dot(gradC.T, np.dot(alpha_tilde_inv, np.dot(gradC, dx) + C))
print(r_true)
print(np.linalg.norm(r_true))


# matrix free
for ii in range(3 * n):
    r[ii] = 0.0

    # M * dx
    mass = M[ii, ii]
    r[ii] += mass * dx[ii]

    # gradC.T * lam
    for j in range(m):  # but only 4 points related to the tetra
        r[ii] += gradC[j, ii] * lam[j]

# gradC.T * alpha_tilde_inv * (gradC * dx + C)
gradC_dx_plus_C = np.zeros((m, 1))
for j in range(m):
    for ii in range(3 * n):
        gradC_dx_plus_C[j] += gradC[j, ii] * dx[ii]
    gradC_dx_plus_C[j] += C[j]

for ii in range(3 * n):
    for j in range(m):
        r[ii] += gradC[j, ii] * alpha_tilde_inv[j, j] * gradC_dx_plus_C[j]

print(r)
print(np.linalg.norm(r))

assert np.linalg.norm(r) - np.linalg.norm(r_true) < 1e-6
