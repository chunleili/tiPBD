import taichi as ti

@ti.kernel
def init_tet_indices(mesh: ti.template(), indices: ti.template()):
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id


def field_from_numpy(x_np):
    import numpy as np
    import taichi as ti
    ti.init()
    x = ti.Vector.field(3, dtype=ti.f32, shape=x_np.shape[0])
    x.from_numpy(x_np)
    return x

def np_to_ti(input, dim=1):
    import numpy as np
    if  isinstance(input, np.ndarray):
        if dim == 1:
            out = ti.field(dtype=ti.f32, shape=input.shape)
            out.from_numpy(input)
        else:
            out = ti.Vector.field(dim, dtype=ti.f32, shape=input.shape)
            out.from_numpy(input)
    else:
        out = input
    return out


@ti.kernel
def random_fill_vec(x: ti.template(), dim: ti.template()):
    shape = x.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for d in ti.static(range(dim)):
                    x[i,j,k][d] = ti.random()

@ti.kernel
def random_fill_scalar(x: ti.template()):
    shape = x.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                    x[i,j,k] = ti.random()


def random_fill(x, dim):
    if dim > 1:
        random_fill_vec(x, dim)
    else:
        random_fill_scalar(x)