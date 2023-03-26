# %%
from sympy import *
from sympy.physics.vector import *
from sympy.physics.mechanics import *
from sympy.abc import *
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# %% 定义 c
sigma_0, sigma_1, sigma_2 = symbols('sigma_0 sigma_1 sigma_2', real=True)
sigma = Matrix([[sigma_0, 0, 0],
                [0, sigma_1, 0],
                [0, 0, sigma_2],
])
sigma

c = (sigma - eye(3)).norm()
c

# %% 第一项：dcdsigma
dcdsigma = zeros(3, 3)

dcdsigma[0, 0] = diff(c, sigma_0)
dcdsigma[1, 1] = diff(c, sigma_1)
dcdsigma[2, 2] = diff(c, sigma_2)
dcdsigma




# %% dsigmadF
dsigmadF = zeros(3, 3)

f00, f01, f02, f10, f11, f12, f20, f21, f22 = symbols('f00 f01 f02 f10 f11 f12 f20 f21 f22', real=True)
F = Matrix([[f00, f01, f02], [f10, f11, f12], [f20, f21, f22]])

U = MatrixSymbol('U', 3, 3)
V = MatrixSymbol('V', 3, 3)

dsigma_df00 = U.transpose() * diff(F, f00) * V
dsigma_df01 = U.transpose() * diff(F, f01) * V
dsigma_df02 = U.transpose() * diff(F, f02) * V
dsigma_df10 = U.transpose() * diff(F, f10) * V
dsigma_df11 = U.transpose() * diff(F, f11) * V
dsigma_df12 = U.transpose() * diff(F, f12) * V
dsigma_df20 = U.transpose() * diff(F, f20) * V
dsigma_df21 = U.transpose() * diff(F, f21) * V
dsigma_df22 = U.transpose() * diff(F, f22) * V

dsigmadF[0,0] = dsigma_df00
dsigmadF[0,1] = dsigma_df01
dsigmadF[0,2] = dsigma_df02
dsigmadF[1,0] = dsigma_df10
dsigmadF[1,1] = dsigma_df11
dsigmadF[1,2] = dsigma_df12
dsigmadF[2,0] = dsigma_df20
dsigmadF[2,1] = dsigma_df21
dsigmadF[2,2] = dsigma_df22
dsigmadF
# %% dFdx
x_0_0, x_0_1, x_0_2, x_1_0, x_1_1, x_1_2, x_2_0, x_2_1, x_2_2 = symbols('x_0_0 x_0_1 x_0_2 x_1_0 x_1_1 x_1_2 x_2_0 x_2_1 x_2_2', real=True)


x_0 = Matrix([x_0_0, x_0_1, x_0_2])
x_1 = Matrix([x_1_0, x_1_1, x_1_2])
x_2 = Matrix([x_2_0, x_2_1, x_2_2])

B = MatrixSymbol('b', 3, 3).as_explicit()

Ds = Matrix([x_1 - x_0 ]).row_join(x_2 - x_0).row_join(x_0)

F = Ds * B

basis = [   [x_0_0],
            [x_0_1],
            [x_0_2],
            [x_1_0],
            [x_1_1],
            [x_1_2],
            [x_2_0], 
            [x_2_1],
            [x_2_2] ]
dFdx = derive_by_array(F, basis)
dFdx


dDsdx = derive_by_array(Ds, basis)
dDsdx
# %% [markdown]
print('最终结果如下')
print('第一项：dcdsigma = ')
dcdsigma
print('第二项：dsigmadF = ')
dsigmadF
print('第三项：dFdx = ')
dFdx

# %%
