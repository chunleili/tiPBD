# %%
from sympy import *
from sympy.abc import *
import sympy
from sympy import init_printing
from sympy import init_session

init_printing()
init_session()

# %%
epsilon_x, epsilon_y, epsilon_xy = symbols("epsilon_x, epsilon_y, epsilon_xy")
C = Matrix([epsilon_x, epsilon_y, epsilon_xy])
C

# %%
alpha = Matrix([[lamda + 2 * mu, lamda, 0], [lamda, lamda + 2 * mu, 0], [0, 0, 2 * mu]])

alpha_inv = alpha.inv()
alpha_inv

# %%
U = C.T * alpha_inv * C
U

# %%
alpha_inv = alpha.inv()
alpha_inv

# %%
U = C.T * alpha_inv * C
U

# %%
ep = Matrix([[epsilon_x, epsilon_xy], [epsilon_xy, epsilon_y]])
ep


# %%
def doubledot(a, b):
    sum = 0
    for i in range(len(a)):
        sum = a[i] * b[i] + sum
    return sum


ep2 = doubledot(ep, ep)
ep2

# %%
psi = mu * ep2 + lamda / 2 * ep.trace() ** 2

expand(psi)

# %%
U
expand(U)

# %%
