# svd.py
import taichi as ti
import taichi.math as tm
ti.init()

@ti.kernel
def test():
    F = ti.Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    F_ = ti.Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    F_[2,2] += 1e-6
    U, S, V = ti.svd(F)
    U_, S_, V_ = ti.svd(F_)
    print("U", U)
    print("S", S)
    print("V", V)
    print("U_", U_)
    print("S_", S_)
    print("V_", V_)

test()