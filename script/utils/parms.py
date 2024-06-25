maxiter = 300
tol=1e-6 # relative tolerance

from collections import namedtuple
Residual = namedtuple('Residual', ['label','r', 't'])