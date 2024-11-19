import taichi as ti
import numpy as np
import sys, os

sys.path.append(os.getcwd())

from engine.collider import *
from engine.visualize import visualize

ti.init()

pos = ti.Vector.field(3, dtype=ti.f32, shape=10000000)


@ti.kernel
def test_sphere():
    cnt = 0
    for i in range(pos.shape[0]):
        pos[i] = [ti.random(), ti.random(), ti.random()]
        pos[i] = pos[i] * 20 - 10
        sdf = sphere(pos[i], 1)
        if sdf > 0:
            pos[i] = [0, 0, 0]
        else:
            cnt += 1
    print(cnt)


@ti.kernel
def test_box():
    cnt = 0
    for i in range(pos.shape[0]):
        pos[i] = [ti.random(), ti.random(), ti.random()]
        pos[i] = pos[i] * 20 - 10
        sdf = box(pos[i], 1)
        if sdf > 0:
            pos[i] = [0, 0, 0]
        else:
            cnt += 1
    print(cnt)


@ti.kernel
def test_torus():
    cnt = 0
    for i in range(pos.shape[0]):
        pos[i] = [ti.random(), ti.random(), ti.random()]
        pos[i] = pos[i] * 20 - 10
        sdf = torus(pos[i], vec2(1, 0.5))
        if sdf > 0:
            pos[i] = [0, 0, 0]
        else:
            cnt += 1
    print(cnt)


if __name__ == "__main__":
    test_sphere()
    visualize(pos, particle_radius_show=0.01)
    test_box()
    visualize(pos, particle_radius_show=0.01)
    test_torus()
    visualize(pos, particle_radius_show=0.01)
