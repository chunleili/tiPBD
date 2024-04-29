import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
import scipy.sparse as sp
from time import time


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
    assert is_pos_def(A)
    return A, b


# https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
def my_cg(A, b, atol=1e-5):
    x=np.zeros(A.shape[0])
    r0=b-A@x
    p=r0.copy()
    r=r0.copy()
    k=0
    while True:
        Ap = A@p
        rTr = r.T@r
        alpha = rTr / (p.T@Ap)
        x1 = x + alpha * p
        r1 = r - alpha * Ap
        if np.linalg.norm(r1)<atol:
            break
        beta=r1.T@r1/(rTr)
        p1=r1+beta*p
        x=x1.copy()
        r=r1.copy()
        p=p1.copy()
        k+=1
    return x1


def test_cg():
    A,b = generate_A_b_psd()

    t=time()
    x_sp, exit_code = cg(A, b, atol=1e-5)
    print("scipy_cg time:", time()-t)
    t=time()
    x_my = my_cg(A, b)
    print("my_cg time:", time()-t)
    print("error:", np.linalg.norm(x_sp-x_my))

# ---------------------------------------------------------------------------- #
#                       preconditioned_conjugate_gradient                      #
# ---------------------------------------------------------------------------- #

def test_pcg():
    A,b = generate_A_b_psd()

    P = sp.diags(1/A.diagonal())

    t=time()
    x_sp, exit_code = cg(A, b, atol=1e-5, M=P)
    print("scipy_cg time:", time()-t)
    t=time()
    x_my = my_pcg(A, b, atol=1e-5, M=P)
    print("my_pcg time:", time()-t)
    print("error:", np.linalg.norm(x_sp-x_my, ord=np.inf))
    print("x(first 5):\n", x_sp[:5],"\n", x_my[:5])


# preconditioned conjugate gradient
# https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
# Note: Based on the scipy(https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/sparse/linalg/_isolve/iterative.py#L406), 
# parameter M is actually the inverse of M in the wiki's formula. We adopt the scipy's definition.
def my_pcg(A, b, atol=1e-5, M=None):
    def solvez(r):
        z = M@r if M is not None else r
        return z
    x=np.zeros(A.shape[0])
    r=b-A@x
    z = solvez(r)
    p=z.copy()
    k=0
    while True:
        Ap = A@p
        rTz = r.T@z
        alpha = r.T@z / (p.T@Ap)
        x1 = x + alpha * p
        r1 = r - alpha * Ap
        if np.linalg.norm(r1)<atol:
            break
        z1 = solvez(r1)
        beta=r1.T@z1/(rTz)
        p1=z1+beta*p
        x=x1.copy()
        r=r1.copy()
        p=p1.copy()
        z=z1.copy()
        k+=1
    return x1


if __name__ == "__main__":
    test_pcg()