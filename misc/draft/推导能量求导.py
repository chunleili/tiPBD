# %%
from sympy import *
from sympy.physics.vector import *
from sympy.physics.mechanics import *
from sympy.abc import *
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %% 定义 c
sigma_0, sigma_1 = symbols("sigma_0 sigma_1", real=True)
sigma = Matrix(
    [
        [sigma_0, 0],
        [0, sigma_1],
    ]
)
sigma

c = (sigma - eye(2)).norm()
c

# %% 第一项：dcdsigma
dcdsigma = zeros(2, 2)

dcdsigma[0, 0] = diff(c, sigma_0)
dcdsigma[1, 1] = diff(c, sigma_1)
dcdsigma


# %% dsigmadF
dsigmadF = zeros(2, 2)

f00, f01, f10, f11 = symbols("f00 f01 f10 f11", real=True)
F = Matrix([[f00, f01], [f10, f11]])

U = MatrixSymbol("U", 2, 2)
V = MatrixSymbol("V", 2, 2)

dsigma_df00 = U.transpose() * diff(F, f00) * V
dsigma_df01 = U.transpose() * diff(F, f01) * V
dsigma_df10 = U.transpose() * diff(F, f10) * V
dsigma_df11 = U.transpose() * diff(F, f11) * V

dsigmadF[0, 0] = dsigma_df00
dsigmadF[0, 1] = dsigma_df01
dsigmadF[1, 0] = dsigma_df10
dsigmadF[1, 1] = dsigma_df11
dsigmadF
# %% dFdx
x_0_0, x_0_1, x_1_0, x_1_1, x_2_0, x_2_1 = symbols("x_0_0 x_0_1 x_1_0 x_1_1 x_2_0 x_2_1", real=True)

x_0 = Matrix([x_0_0, x_0_1])
x_1 = Matrix([x_1_0, x_1_1])
x_2 = Matrix([x_2_0, x_2_1])

B = MatrixSymbol("b", 2, 2).as_explicit()

Ds = Matrix([x_1 - x_0]).row_join(x_2 - x_0)

F = Ds * B

# basis = [   [x_0_0, x_0_1],
#             [x_1_0, x_1_1],
#             [x_2_0, x_2_1]  ]
basis = [[x_0_0], [x_0_1], [x_1_0], [x_1_1], [x_2_0], [x_2_1]]
dFdx = derive_by_array(F, basis)
dFdx


# %% [markdown]
print("最终结果如下")
print("第一项：dcdsigma = ")
dcdsigma
print("第二项：dsigmadF = ")
dsigmadF
print("第三项：dFdx = ")
dFdx
