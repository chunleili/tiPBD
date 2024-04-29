import numpy as np
import scipy.sparse as sp

# judge if A is positive definite
# https://stackoverflow.com/a/44287862/19253199
# if A is symmetric and able to be Cholesky decomposed, then A is positive definite
def is_pos_def(A):
    A=A.toarray()
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            print("A is positive definite")
            return True
        except np.linalg.LinAlgError:
            print("A is not positive definite")
            return False
    else:
        print("A is not positive definite")
        return False

def generate_A_b_psd(n=1000):
    A = sp.random(n, n, density=0.01, format="csr")
    A = A.T @ A
    b = np.random.rand(A.shape[0])
    print(f"Generated PSD A: {A.shape}, b: {b.shape}")
    A = sp.csr_matrix(A)
    return A, b