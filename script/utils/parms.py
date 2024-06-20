maxiter = 150
tol=1e-10 # relative tolerance

from collections import namedtuple
Residual = namedtuple('Residual', ['label','r', 't'])