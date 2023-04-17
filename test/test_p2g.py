import sys,os
import taichi as ti
import numpy as np
sys.path.append(os.getcwd())
from engine.debug_info import debug_info
from engine.p2g import *

def test_p2g_3d():
    ti.init()
    n = 100
    x = ti.Vector.field(3, dtype=ti.f32, shape=n)
    grid_m = ti.field(dtype=ti.f32, shape=(n, n, n))
    np.random.seed(0)
    x_np = np.random.rand(n, 3)
    x.from_numpy(x_np)
    p2g_3d(x, 0.1, grid_m)
    # p2g(x, 0.1, grid_m, 3)
    grid_m_np = debug_info(grid_m,"grid_m")

def test_p2g_2d():
    ti.init()
    n = 100
    x = ti.Vector.field(2, dtype=ti.f32, shape=n)
    grid_m = ti.field(dtype=ti.f32, shape=(n, n))
    np.random.seed(0)
    x_np = np.random.rand(n, 2)
    x.from_numpy(x_np)
    p2g_2d(x, 0.1, grid_m)
    # p2g(x, 0.1, grid_m, 2) 
    grid_m_np = debug_info(grid_m,"grid_m")

if __name__ == '__main__':
    test_p2g_2d()