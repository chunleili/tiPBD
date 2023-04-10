import taichi as ti

vec2 = ti.types.vector(2, dtype=ti.f32)
vec3 = ti.types.vector(3, dtype=ti.f32)

@ti.func
def grad_at_xy(val, x, y, dx, dy)->vec2:
    '''
    Compute the gradient of a 2d scalar field at a given position(x,y).

    Args:
        val (ti.template()): 2D scalar field
        x (ti.f32): x position
        y (ti.f32): y position
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
    
    Returns:
        vec2: gradient at (x,y)

    '''
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j

    grad00 = grad_at_ij(val, i, j, dx, dy)
    grad01 = grad_at_ij(val, i, j+1, dx, dy)
    grad10 = grad_at_ij(val, i+1, j, dx, dy)
    grad11 = grad_at_ij(val, i+1, j+1, dx, dy)
    res = (1-u)*(1-v)*grad00 + u*(1-v)*grad10 + (1-u)*v*grad01 + u*v*grad11
    return res

@ti.func
def grad_at_xyz(val, x, y, z, dx, dy, dz)->vec3:
    '''
    Compute the gradient of a 3d scalar field at a given position(x,y,z).

    Args:
        val (ti.template()): 3D scalar field
        x (ti.f32): x position
        y (ti.f32): y position
        z (ti.f32): z position
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        dz (ti.f32): grid spacing in z direction
    
    Returns:
        vec3: gradient at (x,y,z)
    '''
    i = int(x)
    j = int(y)
    k = int(z)
    u = x - i
    v = y - j
    w = z - k

    grad000 = grad_at_ijk(val, i, j, k, dx, dy, dz)
    grad001 = grad_at_ijk(val, i, j, k+1, dx, dy, dz)
    grad010 = grad_at_ijk(val, i, j+1, k, dx, dy, dz)
    grad011 = grad_at_ijk(val, i, j+1, k+1, dx, dy, dz)
    grad100 = grad_at_ijk(val, i+1, j, k, dx, dy, dz)
    grad101 = grad_at_ijk(val, i+1, j, k+1, dx, dy, dz)
    grad110 = grad_at_ijk(val, i+1, j+1, k, dx, dy, dz)
    grad111 = grad_at_ijk(val, i+1, j+1, k+1, dx, dy, dz)
    res = (1-u)*(1-v)*(1-w)*grad000 + u*(1-v)*(1-w)*grad100 + (1-u)*v*(1-w)*grad010 + u*v*(1-w)*grad110 + (1-u)*(1-v)*w*grad001 + u*(1-v)*w*grad101 + (1-u)*v*w*grad011 + u*v*w*grad111
    return res


def bilinear_weight(x, y):
    '''
    Bilinear sample weights of a 2D scalar field at a given position.
    '''
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j
    return (1-u)*(1-v), u*(1-v), (1-u)*v, u*v


def trilinear_weight(x, y, z):
    '''
    Trilinear sample weights of a 3D scalar field at a given position.
    '''
    i = int(x)
    j = int(y)
    k = int(z)
    u = x - i
    v = y - j
    w = z - k
    return (1-u)*(1-v)*(1-w), u*(1-v)*(1-w), (1-u)*v*(1-w), u*v*(1-w), (1-u)*(1-v)*w, u*(1-v)*w, (1-u)*v*w, u*v*w


def compute_all_gradient(val, dx, dy, dz=None):
    '''
    Compute all the gradient of the SDF field.

    this will automatically determine the dimension of the field and call the corresponding function.
    '''
    shape = val.shape
    dim = len(shape)
    if dim == 2:
        res = compute_all_gradient_2d(val, dx, dy)
    elif dim == 3:
        res = compute_all_gradient_3d(val, dx, dy, dz)
    else:
        raise ValueError(f"Only support 2D and 3D, but got {dim}D")
    return res

        
@ti.kernel
def compute_all_gradient_2d(val:ti.template(), dx: ti.f32, dy: ti.f32, grad:ti.template()):
    '''
    Using central difference to compute all gradients of a 2D scalar field
    
    Args:
        val (ti.template()): 2D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        grad (ti.template()): 2D vector field (result field)
    '''
    shape = val.shape
    for i, j in ti.ndrange((1, shape[0]-1), (1, shape[1]-1)):
        grad[i, j] = ti.Vector([(val[i+1, j] - val[i-1, j]/dx), (val[i, j+1] - val[i, j-1])/dy]) * 0.5 

    for i in range(1, shape[0]-1):
        grad[i, 0] = ti.Vector([
                (val[i+1, 0] - val[i-1, 0]/dx),
                (val[i  , 1] - val[i  , 0])/dy
                ])
        grad[i, shape[1]-1] = ti.Vector([
                (val[i+1, shape[1]-1] - val[i-1, shape[1]-1]/dx) * 0.5,
                (val[i  , shape[1]-1] - val[i  , shape[1]-2])/dy
                ])
    
    for j in range(1, shape[1]-1):
        grad[0, j] = ti.Vector([
            (val[1, j  ] - val[0, j]/dx),
            (val[0, j+1] - val[0, j-1])/dy * 0.5
            ])
        grad[shape[0]-1, j] = ti.Vector([
            (val[shape[0]-1, j  ] - val[shape[0]-2, j]/dx),
            (val[shape[0]-1, j+1] - val[shape[0]-1, j-1])/dy * 0.5
            ])


@ti.kernel
def compute_all_gradient_3d(val:ti.template(), dx: ti.f32, dy: ti.f32, dz: ti.f32, grad:ti.template()):
    '''
    Using central difference to compute all gradients of a 3D scalar field

    Args:
        val (ti.template()): 3D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        dz (ti.f32): grid spacing in z direction
        grad (ti.template()): 3D vector field (result field)
    '''
    shape = val.shape
    for i, j, k in ti.ndrange((1, shape[0]-2), (1, shape[1]-2), (1, shape[2]-2)):
        grad[i, j, k] = ti.Vector([
            (val[i+1, j, k] - val[i-1, j, k])/dx,
            (val[i, j+1, k] - val[i, j-1, k])/dy,
            (val[i, j, k+1] - val[i, j, k-1])/dz
            ]) * 0.5
    for i in range(1, shape[0]-2):
        for j in range(1, shape[1]-2):
            grad[i, j, 0] = ti.Vector([
                (val[i+1, j, 0] - val[i-1, j, 0])/dx * 0.5,
                (val[i, j+1, 0] - val[i, j-1, 0])/dy * 0.5,
                (val[i, j, 1] - val[i, j, 0])/dz
                ])
            grad[i, j, shape[2]-1] = ti.Vector([
                (val[i+1, j, shape[2]-1] - val[i-1, j, shape[2]-1])/dx * 0.5,
                (val[i, j+1, shape[2]-1] - val[i, j-1, shape[2]-1])/dy * 0.5,
                (val[i, j, shape[2]-1] - val[i, j, shape[2]-2])/dz
                ])
    for i in range(1, shape[0]-2):
        for k in range(1, shape[2]-2):
            grad[i, 0, k] = ti.Vector([
                (val[i+1, 0, k] - val[i-1, 0, k])/dx * 0.5,
                (val[i, 1, k] - val[i, 0, k])/dy,
                (val[i, 0, k+1] - val[i, 0, k-1])/dz * 0.5
                ])
            grad[i, shape[1]-1, k] = ti.Vector([
                (val[i+1, shape[1]-1, k] - val[i-1, shape[1]-1, k])/dx * 0.5,
                (val[i, shape[1]-1, k] - val[i, shape[1]-2, k])/dy,
                (val[i, shape[1]-1, k+1] - val[i, shape[1]-1, k-1])/dz * 0.5
                ])
    for j in range(1, shape[1]-2):
        for k in range(1, shape[2]-2):
            grad[0, j, k] = ti.Vector([
                (val[1, j, k] - val[0, j, k])/dx,
                (val[0, j+1, k] - val[0, j-1, k])/dy * 0.5,
                (val[0, j, k+1] - val[0, j, k-1])/dz * 0.5
                ])
            grad[shape[0]-1, j, k] = ti.Vector([
                (val[shape[0]-1, j, k] - val[shape[0]-2, j, k])/dx,
                (val[shape[0]-1, j+1, k] - val[shape[0]-1, j-1, k])/dy * 0.5,
                (val[shape[0]-1, j, k+1] - val[shape[0]-1, j, k-1])/dz * 0.5
                ])

@ti.func
def grad_at_ij(val:ti.template(), dx: ti.f32, dy: ti.f32, i: ti.i32, j: ti.i32)->vec2:
    '''
    Using central difference to compute the gradient of a 2D scalar field at a given point

    Args:
        val (ti.template()): 2D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        i (ti.i32): x index of the point
        j (ti.i32): y index of the point

    Returns:
        vec2: gradient at the given point
    '''
    shape = val.shape
    res = ti.Vector([0.0, 0.0])
    if i == 0:
        res[0] = (val[1, j] - val[0, j])/dx
    elif i == shape[0]-1:
        res[0] = (val[shape[0]-1, j] - val[shape[0]-2, j])/dx
    else:
        res[0] = (val[i+1, j] - val[i-1, j])/dx * 0.5
    if j == 0:
        res[1] = (val[i, 1] - val[i, 0])/dy
    elif j == shape[1]-1:
        res[1] = (val[i, shape[1]-1] - val[i, shape[1]-2])/dy
    else:
        res[1] = (val[i, j+1] - val[i, j-1])/dy * 0.5
    return res


@ti.func
def grad_at_ijk(val:ti.template(), dx: ti.f32, dy: ti.f32, dz: ti.f32, i: ti.i32, j: ti.i32, k: ti.i32)->vec3:
    '''
    Using central difference to compute the gradient of a 3D scalar field at a given point

    Args:
        val (ti.template()): 3D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        dz (ti.f32): grid spacing in z direction
        i (ti.i32): x index of the point
        j (ti.i32): y index of the point
        k (ti.i32): z index of the point

    Returns:
        vec3: gradient at the given point
    '''
    shape = val.shape
    res = ti.Vector([0.0, 0.0, 0.0])
    if i == 0:
        res[0] = (val[1, j, k] - val[0, j, k])/dx
    elif i == shape[0]-1:
        res[0] = (val[shape[0]-1, j, k] - val[shape[0]-2, j, k])/dx
    else:
        res[0] = (val[i+1, j, k] - val[i-1, j, k])/dx * 0.5
    if j == 0:
        res[1] = (val[i, 1, k] - val[i, 0, k])/dy
    elif j == shape[1]-1:
        res[1] = (val[i, shape[1]-1, k] - val[i, shape[1]-2, k])/dy
    else:
        res[1] = (val[i, j+1, k] - val[i, j-1, k])/dy * 0.5
    if k == 0:
        res[2] = (val[i, j, 1] - val[i, j, 0])/dz
    elif k == shape[2]-1:
        res[2] = (val[i, j, shape[2]-1] - val[i, j, shape[2]-2])/dz
    else:
        res[2] = (val[i, j, k+1] - val[i, j, k-1])/dz * 0.5


@ti.func
def bilinear_interpolate(val:ti.template(), x: ti.f32, y: ti.f32)->ti.f32:
    '''
    Bilinear interpolation of a 2D scalar field

    Args:
        val (ti.template()): 2D scalar field
        x (ti.f32): x coordinate of the point
        y (ti.f32): y coordinate of the point

    Returns:
        ti.f32: interpolated value
    '''
    shape = val.shape
    i = int(x)
    j = int(y)
    if i < 0 or i >= shape[0]-1 or j < 0 or j >= shape[1]-1:
        return 0.0
    s = x - i
    t = y - j
    return (1-s)*(1-t)*val[i, j] + s*(1-t)*val[i+1, j] + (1-s)*t*val[i, j+1] + s*t*val[i+1, j+1]

@ti.func
def trilinear_interpolate(val:ti.template(), x: ti.f32, y: ti.f32, z: ti.f32)->ti.f32:
    '''
    Trilinear interpolation of a 3D scalar field

    Args:
        val (ti.template()): 3D scalar field
        x (ti.f32): x coordinate of the point
        y (ti.f32): y coordinate of the point
        z (ti.f32): z coordinate of the point

    Returns:
        ti.f32: interpolated value
    '''
    shape = val.shape
    i = int(x)
    j = int(y)
    k = int(z)
    if i < 0 or i >= shape[0]-1 or j < 0 or j >= shape[1]-1 or k < 0 or k >= shape[2]-1:
        return 0.0
    s = x - i
    t = y - j
    u = z - k
    return (1-s)*(1-t)*(1-u)*val[i, j, k] + s*(1-t)*(1-u)*val[i+1, j, k] + (1-s)*t*(1-u)*val[i, j+1, k] + s*t*(1-u)*val[i+1, j+1, k] + (1-s)*(1-t)*u*val[i, j, k+1] + s*(1-t)*u*val[i+1, j, k+1] + (1-s)*t*u*val[i, j+1, k+1] + s*t*u*val[i+1, j+1, k+1]




def test_grad_at_ij():
    '''
    Test the gradient function in 2D
    '''
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



if __name__ == '__main__':
    ti.init()
    test_grad_at_ij()