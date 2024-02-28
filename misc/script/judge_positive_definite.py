import scipy
import numpy as np
def is_pos_def(x):
    return scipy.linalg.ishermitian(x)

A = scipy.io.mmread("./result/test/A_f100.mtx")
A=A.todense()
chol_A = np.linalg.cholesky(A)
print(chol_A)