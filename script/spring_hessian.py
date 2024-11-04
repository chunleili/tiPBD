"""
Sample code automatically generated on 2024-11-04 09:30:17

by www.matrixcalculus.org

from input

d/dp (1-l0/norm2(p-q))*(p-q) = l0/norm2(p-q).^3*(p-q)*(p-q)'+(1-l0/norm2(p-q))*eye

where

l0 is a scalar
p is a vector
q is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(l0, p, q):
    if isinstance(l0, np.ndarray):
        dim = l0.shape
        assert dim == (1, )
    assert isinstance(p, np.ndarray)
    dim = p.shape
    assert len(dim) == 1
    p_rows = dim[0]
    assert isinstance(q, np.ndarray)
    dim = q.shape
    assert len(dim) == 1
    q_rows = dim[0]
    assert q_rows == p_rows

    t_0 = (p - q)
    t_1 = np.linalg.norm(t_0)
    t_2 = (1 - (l0 / t_1))
    functionValue = (t_2 * t_0)
    gradient = (((l0 / (t_1 ** 3)) * np.outer(t_0, t_0)) + (t_2 * np.eye(q_rows, p_rows)))

    return functionValue, gradient

def checkGradient(l0, p, q, t=1E-6):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    # t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(l0, p + t * delta, q)
    f2, _ = fAndG(l0, p - t * delta, q)
    f, g = fAndG(l0, p, q)
    err = np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1))
    print('error of t=', t, ' = ',
          err)
    return err

def generateRandomData():
    l0 = np.random.randn(1)
    p = np.random.randn(3)
    q = np.random.randn(3)

    return l0, p, q

if __name__ == '__main__':
    l0, p, q = generateRandomData()
    functionValue, gradient = fAndG(l0, p, q)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    err = []
    ts = [1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6]
    for t in ts:
        err.append(checkGradient(l0, p, q, t))
    

    import matplotlib.pyplot as plt
    plt.plot(-np.log(ts), -np.log(err))
    plt.show()