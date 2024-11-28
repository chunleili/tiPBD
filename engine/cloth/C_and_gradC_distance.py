import taichi as ti
import numpy as np
import scipy

def compute_C_and_gradC_distance(pos, vert, rest_len):
    """Compute C and gradC for distance constraint
    
    Parameters
    ----------
        pos: taichi vector field, shape = NV, n=3.
            positions

        vert:  taichi vector field, shape = NCONS, n=2
            vertex indices of constraints.

        rest_len: taichi field, shape=NCONS.
            rest lengths

    Returns
    ----------
        C:  taichi field, shape= NCONS
            constraints

        gradC: taichi vector field, shape=(NCONS,2), n=3
            constraint gradient. Stored in array form instead of sparse matrix.

    Notes
    -----
    - vector C is in R^m. For one constraint j:

        C_j = l_ij - l_0

    - sparse matrix nabla C is in R^(m x 3n). For one constraint j:

        nabla C_j = [g, -g], where g = (p-q)/l is 3x1 vector.

        One constraint j corresponds to one row of sparse matrix nabla C.

        The first nonzero value g locates at j row, 3*i1/3*i1+1/3*i1+2 columns;

        The second nonzero value -g locates at j row, 3*i2/3*i2+1/3*i2+2 columns

    """

    constraints = ti.field(dtype=ti.f32, shape=vert.shape[0])
    gradC = ti.Vector.field(3, dtype=ti.f32, shape=(vert.shape[0],2))
    
    @ti.kernel
    def compute_C_and_gradC_kernel(
        pos:ti.template(),
        gradC: ti.template(),
        vert:ti.template(),
        constraints:ti.template(),
        rest_len:ti.template(),
    ):
        for i in range(vert.shape[0]):
            idx0, idx1 = vert[i]
            dis = pos[idx0] - pos[idx1]
            lij = dis.norm()
            if lij == 0.0:
                continue
            constraints[i] = lij - rest_len[i]
            if constraints[i] < 1e-6:
                continue
            g = dis.normalized()

            gradC[i, 0] += g
            gradC[i, 1] += -g
            
    compute_C_and_gradC_kernel(pos,
                                gradC,
                                vert,
                                constraints,
                                rest_len)
    assert not np.any(np.isnan(constraints.to_numpy()))
    assert not np.any(np.isnan(gradC.to_numpy()))
    return constraints, gradC


def fill_G_distance(gradC, vert, NV):
    """Fill dense array form gradC into a sparse matrix G.

    Parameters
    ----------
        gradC: taichi vector field, shape=(NCONS,2), n=3
            constraint gradient. Stored in array form instead of sparse matrix.

        vert:  taichi vector field, shape = NCONS, n=2
            vertex indices of constraints.

        NV: int
            number of vertices

    Returns
    ----------
        G: scipy csr_matrix
            sparse matrix of constraint gradient.


    Note
    ---------
        - sparse matrix nabla C is in R^(m x 3n). For one constraint j:

        nabla C_j = [g, -g], where g = (p-q)/l is 3x1 vector.

        One constraint j corresponds to one row of sparse matrix nabla C.

        The first nonzero value g locates at j row, 3*i1/3*i1+1/3*i1+2 columns;

        The second nonzero value -g locates at j row, 3*i2/3*i2+1/3*i2+2 columns
    """

    NCONS = vert.shape[0]

    MAX_NNZ =NCONS * 6

    @ti.kernel
    def fill_gradC_triplets_kernel(
        ii:ti.types.ndarray(dtype=ti.i32),
        jj:ti.types.ndarray(dtype=ti.i32),
        vv:ti.types.ndarray(dtype=ti.f32),
        gradC: ti.template(),
        vert: ti.template(),
    ):
        cnt=0
        ti.loop_config(serialize=True)
        for j in range(vert.shape[0]):
            ind = vert[j]
            for p in range(2):
                for d in range(3):
                    i = ind[p]
                    ii[cnt],jj[cnt],vv[cnt] = j, 3 * i + d, gradC[j, p][d]
                    cnt+=1
    
    G_ii, G_jj, G_vv = np.zeros(MAX_NNZ, dtype=np.int32), np.zeros(MAX_NNZ, dtype=np.int32), np.zeros(MAX_NNZ, dtype=np.float32)
    assert not np.any(np.isnan(gradC.to_numpy()))
    assert not np.any(np.isnan(G_vv))
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, vert)
    G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(NCONS, 3*NV))
    return G