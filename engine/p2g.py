import taichi as ti
@ti.kernel
def p2g_2d(x: ti.template(), dx:ti.f32, grid_m:ti.template()):
    '''
    将周围粒子的质量scatter到2D网格上。实际上，只要替换grid_m, 可以scatter任何标量场。

    Args:
        x (ti.template()): 粒子位置
        dx (ti.f32): 网格间距
        grid_m (ti.template()): 网格质量(输出)
    '''
    inv_dx = 1.0 / dx
    p_mass = 1.0
    for p in x:
        base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                I = ti.Vector([i, j])
                weight = w[i].x * w[j].y
                if in_bound(base, I, grid_m.shape):
                    grid_m[base + I] += weight * p_mass


@ti.kernel
def p2g_3d(x: ti.template(), dx:ti.f32, grid_m:ti.template()):
    '''
    将周围粒子的质量scatter到3D网格上。实际上，只要替换grid_m, 可以scatter任何标量场。

    Args:
        x (ti.template()): 粒子位置
        dx (ti.f32): 网格间距
        grid_m (ti.template()): 网格质量(输出)
    '''
    inv_dx = 1.0 / dx
    p_mass = 1.0
    shape = grid_m.shape
    for p in x:
        base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                        I = ti.Vector([i, j, k])
                        weight = w[i].x * w[j].y * w[k].z
                        if in_bound(base, I, shape):
                            grid_m[base + I] += weight * p_mass

@ti.func 
def in_bound(index, offset, shape):
    res = True
    for i in ti.static(range(len(shape))):
        if index[i] + offset[i]  < 0 or index[i] + offset[i] >= shape[i] :
            res = False
    return res


@ti.kernel
def p2g(x: ti.template(), dx:ti.f32, grid_m:ti.template(), dim:ti.template()):
    '''
    将周围粒子的质量scatter到网格上。实际上，只要替换grid_m, 可以scatter任何标量场。
    (此函数相比p2g_2d与pg2_3d来说，增加了参数dim)
    Args:
        x (ti.template()): 粒子位置
        dx (ti.f32): 网格间距
        grid_m (ti.template()): 网格质量(输出)
        dim (ti.template()): 网格维度
    '''
    inv_dx = 1.0 / dx
    p_mass = 1.0
    for p in x:
        base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # Loop over 3x3 grid node neighborhood
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]
            if in_bound(base, offset, grid_m.shape):
                grid_m[base + offset] += weight * p_mass


def test_p2g_3d():
    import numpy as np
    import taichi as ti
    from debug_info import debug_info
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
    import numpy as np
    import taichi as ti
    from debug_info import debug_info
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