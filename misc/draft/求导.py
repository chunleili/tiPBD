import taichi as ti
import meshtaichi_patcher as patcher
from taichi.lang.ops import sqrt
import taichi.math as tm
import numpy as np


def computeGradient(B, U, S, V):
    # Compute (dsdF)
    u00, u01, u02 = U[0, 0], U[0, 1], U[0, 2]
    u10, u11, u12 = U[1, 0], U[1, 1], U[1, 2]
    u20, u21, u22 = U[2, 0], U[2, 1], U[2, 2]
    v00, v01, v02 = V[0, 0], V[0, 1], V[0, 2]
    v10, v11, v12 = V[1, 0], V[1, 1], V[1, 2]
    v20, v21, v22 = V[2, 0], V[2, 1], V[2, 2]
    dsdF00 = np.array([u00 * v00, u01 * v01, u02 * v02])
    dsdF10 = np.array([u10 * v00, u11 * v01, u12 * v02])
    dsdF20 = np.array([u20 * v00, u21 * v01, u22 * v02])
    dsdF01 = np.array([u00 * v10, u01 * v11, u02 * v12])
    dsdF11 = np.array([u10 * v10, u11 * v11, u12 * v12])
    dsdF21 = np.array([u20 * v10, u21 * v11, u22 * v12])
    dsdF02 = np.array([u00 * v20, u01 * v21, u02 * v22])
    dsdF12 = np.array([u10 * v20, u11 * v21, u12 * v22])
    dsdF22 = np.array([u20 * v20, u21 * v21, u22 * v22])
    print("dsdF00:", dsdF00)
    print("dsdF10:", dsdF10)
    print("dsdF20:", dsdF20)
    print("dsdF01:", dsdF01)
    print("dsdF11:", dsdF11)
    print("dsdF21:", dsdF21)
    print("dsdF02:", dsdF02)
    print("dsdF12:", dsdF12)
    print("dsdF22:", dsdF22)

    dsdF = np.zeros((3, 3, 3, 3))
    mid = np.mat([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(3):
        for j in range(3):
            mid[i, j] = 1
            dsdF[i, j] = U.transpose() @ mid @ V
            print(i, j, ":", dsdF[i, j])
    pass


F = np.mat([np.random.rand(3) for i in range(3)])
U, S, V = np.linalg.svd(F)

B = np.mat([np.random.rand(3) for i in range(3)])
computeGradient(B, U, S, V)


ti.init()
x = ti.field(dtype=ti.f32, shape=(3, 3, 3), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def compute_y():
    for i, j, k in ti.ndrange(3, 3, 3):
        y[None] += x[i, j, k]


for i, j, k in np.ndindex((3, 3, 3)):
    x[i, j, k] = F[i, j] * S[k]

dt = 0.01


@ti.kernel
def advance():
    for i, j, k in x:
        x[i, j, k] += dt * 0.01 * x.grad[i, j, k]


def substep():
    with ti.ad.Tape(y):
        compute_y()
    advance()


for i in range(10):
    substep()

for i, j, k in np.ndindex((3, 3, 3)):
    print(i, j, k, "dy/dx =", x.grad[i, j, k], " at x =", x[i, j, k])
