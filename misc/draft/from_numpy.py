import taichi as ti
import numpy as np

ti.init()
mat4x4 = [[0] * 4 for _ in range(4)]
proj_ti = ti.Matrix(mat4x4, dt=ti.f32)
proj = np.random.rand(4, 4)


def fill4x4(mat_ti, mat_np):
    for i in range(4):
        for j in range(4):
            mat_ti[i, j] = mat_np[i, j]


fill4x4(proj_ti, proj)
print(proj_ti)
