# https://github.com/ddemidov/amgcl/issues/135#issuecomment-562911575
# // https://github.com/petsc/petsc/blob/6dd80bdec2980634f66887e5e77bea5a1e70dedf/src/mat/interface/matnull.c#L127
# https://scicomp.stackexchange.com/questions/33017/null-space-for-smoothed-aggregation-algebraic-multigrid/34155#34155

from scipy.io import mmread, mmwrite
from numpy import zeros, sqrt, dot, linalg


def mesh_to_coo(readpath):
    import meshio
    mesh = meshio.read(readpath)
    coo = mesh.points
    # coo = coo.ravel(order='F').reshape(-1,1)
    return coo

def calc_RBM2d():
    N = mesh_to_coo('data/model/bunnyBig/bunnyBig.node')

    # Guessing the signs of the coordinates
    # (rotational vector from petsc uses opposite signs):
    x = -N[1::2].flatten()
    y =  N[0::2].flatten()

    n = N.shape[0]

    N = zeros((n,3))

    # First two vectors:
    N[0::2,0] = 1 / sqrt(n)
    N[1::2,1] = 1 / sqrt(n)

    # Third, rotational vector:
    N[0::2,2] = -y
    N[1::2,2] =  x

    # Orthonormalize the rotational vector w.r.t. the first two:
    N[:,2] -= dot(N[:,2], N[:,0]) * N[:,0] + dot(N[:,2], N[:,1]) * N[:,1]
    N[:,2] /= linalg.norm(N[:,2])
 
    return N


def calc_RBM3d():
    coo = mesh_to_coo('data/model/bunnyBig/bunnyBig.node')
    coo = coo.flatten()
    x = coo[0::3]
    y = coo[1::3]
    z = coo[2::3]

    v = zeros((coo.shape[0],6))

    n = coo.shape[0]/3
    v[0::3, 0] = 1./sqrt(n)
    v[1::3, 1] = 1./sqrt(n)
    v[0::3, 2] = 1./sqrt(n)

    # v[0::3, 3] = -z
    # v[1::3, 3] = y

    # v[0::3, 4] = z
    # v[1::3, 4] = -x

    # v[0::3, 5] = -y
    # v[1::3, 5] = x

    v[0::3, 3] = y
    v[1::3, 3] = -x

    v[1::3, 4] = -z
    v[2::3, 4] = y

    v[0::3, 5] = z
    v[2::3, 5] = -x


    # orthonormalize
    for i in range(3, 6):
        for j in range(0, i):
            v[:, i] -= dot(v[:, i], v[:, j]) * v[:, j]
        v[:, i] /= linalg.norm(v[:, i])

    return v

