# %% 导入模块
from sympy import *
from sympy.physics.vector import *
from sympy.physics.mechanics import *
from sympy.abc import *
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %% 初始化
# 速度
u = symbols("u", real=True)
# kinematic map
G = symbols("G", real=True)
# 广义坐标
q = symbols("q", real=True)
# 时间步长
dt = symbols("dt", real=True)

q = G * u * dt

U = symbols("U", real=True)
C = symbols("C", real=True)

U = (1 / 2) * k * C**2

f = symbols("f", real=True)
f = G * U.diff(q)
f
# %%
