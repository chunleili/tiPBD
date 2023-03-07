# %%
from sympy import *
from sympy.physics.vector import *
from sympy.physics.mechanics import *
from sympy.abc import *


# %% 定义energy
I = Identity(3)
I
sigma_0, sigma_1, sigma_2 = symbols('sigma_0 sigma_1 sigma_2')
sigma = Matrix([[sigma_0, 0, 0],
                [0, sigma_1, 0],
                [0, 0, sigma_2]])
sigma
energy_mat = sigma - I
energy_mat = energy_mat.as_explicit()
energy_mat
energy = energy_mat.norm()
energy
# %% 定义Ds 和 Dm.T即B 以及F
x_0 = Matrix(['x_0_0', 'x_0_1'])
x_1 = Matrix(['x_1_0', 'x_1_1'])
x_2 = Matrix(['x_2_0', 'x_2_1'])
x_1 - x_0
x_2 - x_0
Ds = (x_1 - x_0).row_join(x_2 - x_1)
Ds

X_0 = Matrix(['X_0_0', 'X_0_1'])
X_1 = Matrix(['X_1_0', 'X_1_1'])
X_2 = Matrix(['X_2_0', 'X_2_1'])
X_1 - X_0
X_2 - X_0
Dm = (X_1 - X_0).row_join(X_2 - X_1)
Dm

B = Dm.inv()
B

F = Ds * B
F

# %% 
# assume sigma_0 is real
assuming(sigma_0, 'real')
assuming(sigma_1, 'real')
assuming(sigma_2, 'real')
diff(energy, sigma_0)
diff(energy, sigma_1)
diff(energy, sigma_2)
