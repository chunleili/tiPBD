import numpy as np


def compute_bary_coord_batch(node_coordinates, node_ids, p):
    """
    Find which element of a mesh contains a given point.

    相比于compute函数，避免两个for循环，加速计算。

    https://stackoverflow.com/a/57901916

    Args:
        node_coordinates: (n_nodes,3) array containing the coordinates of each node
        node_ids: (n_tet, 4) array, where the i-th row gives the vertex indices of the i-th tetrahedron.
    """
    ori = node_coordinates[node_ids[:, 0], :]
    v1 = node_coordinates[node_ids[:, 1], :] - ori
    v2 = node_coordinates[node_ids[:, 2], :] - ori
    v3 = node_coordinates[node_ids[:, 3], :] - ori
    n_tet = len(node_ids)
    v1r = v1.T.reshape((3, 1, n_tet))
    v2r = v2.T.reshape((3, 1, n_tet))
    v3r = v3.T.reshape((3, 1, n_tet))
    mat = np.concatenate((v1r, v2r, v3r), axis=1)
    inv_mat = np.linalg.inv(mat.T).T  # https://stackoverflow.com/a/41851137/12056867
    if p.size == 3:
        p = p.reshape((1, 3))
    n_p = p.shape[0]
    orir = np.repeat(ori[:, :, np.newaxis], n_p, axis=2)
    newp = np.einsum("imk,kmj->kij", inv_mat, p.T - orir)
    val = np.all(newp >= 0, axis=1) & np.all(newp <= 1, axis=1) & (np.sum(newp, axis=1) <= 1)
    id_tet, id_p = np.nonzero(val)
    res = -np.ones(n_p, dtype=id_tet.dtype)  # Sentinel value
    res[id_p] = id_tet
    return res
