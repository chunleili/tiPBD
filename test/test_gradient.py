import sys, os
import taichi as ti

sys.path.append(os.getcwd())
from engine.gradient import *


def test_grad_at_ij():
    """
    Test the gradient function in 2D
    """
    val = ti.field(ti.f32, shape=(3, 3))
    val[0, 0] = 0.0
    val[1, 0] = 1.0
    val[2, 0] = 2.0
    val[0, 1] = 3.0
    val[1, 1] = 4.0
    val[2, 1] = 5.0
    val[0, 2] = 6.0
    val[1, 2] = 7.0
    val[2, 2] = 8.0
    dx = 1.0
    dy = 1.0

    @ti.kernel
    def test():
        for i in range(3):
            for j in range(3):
                print(grad_at_ij(val, dx, dy, i, j))

    test()


if __name__ == "__main__":
    ti.init()
    test_grad_at_ij()
