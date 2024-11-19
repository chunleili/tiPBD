
import numpy as np
import scipy.sparse

# https://en.wikipedia.org/wiki/Chebyshev_iteration
def SolChebyshev002(A, b, x0, iterNum, lMax, lMin):
    d = (lMax + lMin) / 2
    c = (lMax - lMin) / 2
    # preCond = np.eye(A.shape[0])  # 预处理矩阵
    preCond = scipy.sparse.eye(A.shape[0]).tocsr()  # 预处理矩阵
    x = x0
    r = b - A@x

    for i in range(1, iterNum + 1):
        # z = scipy.sparse.linalg.spsolve(preCond, r)
        z=r.copy()
        if i == 1:
            p = z
            alpha = 1 / d
        elif i == 2:
            beta = (1 / 2) * (c * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            p = z + beta * p
        else:
            beta = (c * alpha / 2) ** 2
            alpha = 1 / (d - beta / alpha)
            p = z + beta * p

        x = x + alpha * p
        r = b - A@x
        if np.linalg.norm(r) < 1e-15:
            break
    if i == iterNum:
        print("Chebyshev Method not converged in ", iterNum, " iterations")

    return x

from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial

def calc_spectral_radius(A):
    spectral_radius = approximate_spectral_radius(A) # legacy python version
    return spectral_radius

def setup_chebyshev(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3):
    """Set up Chebyshev."""
    rho = calc_spectral_radius(A)
    a = rho * lower_bound
    b = rho * upper_bound
    chebyshev_coeff = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]
    return a, b, chebyshev_coeff

# from cloth3d import chebyshev_polynomial_coefficients, setup_chebyshev, chebyshev

def chebyshev(A, x, b, coefficients, iterations=1):
    x = np.ravel(x)
    b = np.ravel(b)
    for _i in range(iterations):
        residual = b - A*x
        h = coefficients[0]*residual
        for c in coefficients[1:]:
            h = c*residual + A*h
        x += h


def test():
    path = "E:/Dev/tiPBD/result/latest/A/"
    A = scipy.sparse.load_npz(path+"A_F0-0.npz") # load
    b = np.load(path+"b_F0-0.npy")
    x = np.zeros(b.shape)
    maxIter = 10
    a,bb,chebyshev_coeff  = setup_chebyshev(A)
    print(np.linalg.norm(b - A*x))
    chebyshev(A, x, b, chebyshev_coeff, maxIter)
    print(x)
    print(np.linalg.norm(b - A*x))

    x = np.zeros(b.shape)
    SolChebyshev002(A, b, x, maxIter, a, bb)
    print(x)
    print(np.linalg.norm(b - A*x))

test()