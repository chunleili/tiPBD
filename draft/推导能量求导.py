# %%
from sympy import *
from sympy.physics.vector import *
from sympy.physics.mechanics import *
from sympy.abc import *


# %% 定义sigma
sigma_0, sigma_1 = symbols('sigma_0 sigma_1', real=True)
sigma = Matrix([[sigma_0, 0],
                [0, sigma_1],
])
sigma

# %% 定义能量
energy_mat = sigma - eye(2)
energy_mat
# %% 
energy = energy_mat.norm()
c = energy
c
# %% 求导
diff(c, sigma_0)
# %%
diff(c, sigma_1)

# %% 
# F_ = MatrixSymbol('F', 2, 2)
# U = MatrixSymbol('U', 2, 2)
# V = MatrixSymbol('V', 2, 2)
# U
# V
# F_ = U * sigma * V.T
# F_
# sigma.diff(F_[i,j])



# %% 

# # %% 
# x = Matrix([['x_0_0', 'x_0_1'], ['x_1_0', 'x_1_1'], ['x_2_0', 'x_2_1']], real = True).transpose()
# x

# # %%
# x_0_0 = symbols('x_0_0', real=True)
# x_0_1 = symbols('x_0_1', real=True)
# x_1_0 = symbols('x_1_0', real=True)
# x_1_1 = symbols('x_1_1', real=True)
# x_0_0
# %% 定义Ds
x_0 = Matrix(['x_0_0', 'x_0_1'],real=True)
x_1 = Matrix(['x_1_0', 'x_1_1'],real=True)
x_2 = Matrix(['x_2_0', 'x_2_1'],real=True)

x_0
# %%
Ds = (x_1 - x_0).row_join(x_2 - x_1)
Ds

# # %%
# X_0 = Matrix(['X_0_0', 'X_0_1'],real=True)
# X_1 = Matrix(['X_1_0', 'X_1_1'],real=True)
# X_2 = Matrix(['X_2_0', 'X_2_1'],real=True)
# X_1 - X_0
# X_2 - X_0
# Dm = (X_1 - X_0).row_join(X_2 - X_1)
# Dm

# # %%
# B = Dm.inv()
# B

# %%
B = MatrixSymbol('B', 2, 2).as_explicit()
B

# %%
F = Ds * B
F

# %%
diff(F, 'x_0_0')

# %%
diff(F, x_0)
# %%
diff(F, x_1)

# %%
diff(F, x_2)

# %%
