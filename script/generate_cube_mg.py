# source: https://github.com/FantasyVR/amg_pbd/blob/46b8907275839d86d8cff690c00a01520904f969/arap_xpbd/arap_3d/generate_coarse_fine_mesh.py

import numpy as np
import meshio

# Usage:
# # original size: 0.1 and 0.5
# fine_points, fine_tet_indices, fine_tri_indices = generate_mesh(2.0, 0.1)
# write_tet("data/model/cube/fine_new.node", fine_points, fine_tet_indices)
# coarse_points, coarse_tet_indices, coarse_tri_indices = generate_mesh(2.0, 0.5)
# write_tet("data/model/cube/coarse_new.node", coarse_points, coarse_tet_indices)
def generate_cube_mesh(len, grid_dx=0.1):
    num_grid = int(len // grid_dx)
    points = np.zeros(((num_grid + 1) ** 3, 3), dtype=float)
    for i in range(num_grid + 1):
        for j in range(num_grid + 1):
            for k in range(num_grid + 1):
                points[i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k] = [i * grid_dx, j * grid_dx, k * grid_dx]
    tet_indices = np.zeros(((num_grid) ** 3 * 5, 4), dtype=int)
    tri_indices = np.zeros(((num_grid) ** 3 * 12, 3), dtype=int)
    for i in range(num_grid):
        for j in range(num_grid):
            for k in range(num_grid):
                id0 = i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k
                id1 = i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k + 1
                id2 = i * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k
                id3 = i * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k + 1
                id4 = (i + 1) * (num_grid + 1) ** 2 + j * (num_grid + 1) + k
                id5 = (i + 1) * (num_grid + 1) ** 2 + j * (num_grid + 1) + k + 1
                id6 = (i + 1) * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k
                id7 = (i + 1) * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k + 1
                tet_start = (i * num_grid**2 + j * num_grid + k) * 5
                tet_indices[tet_start] = [id0, id1, id2, id4]
                tet_indices[tet_start + 1] = [id1, id4, id5, id7]
                tet_indices[tet_start + 2] = [id2, id4, id6, id7]
                tet_indices[tet_start + 3] = [id1, id2, id3, id7]
                tet_indices[tet_start + 4] = [id1, id2, id4, id7]
                tri_start = (i * num_grid**2 + j * num_grid + k) * 12
                tri_indices[tri_start] = [id0, id2, id4]
                tri_indices[tri_start + 1] = [id2, id4, id6]
                tri_indices[tri_start + 2] = [id0, id1, id2]
                tri_indices[tri_start + 3] = [id1, id2, id3]
                tri_indices[tri_start + 4] = [id0, id1, id4]
                tri_indices[tri_start + 5] = [id1, id4, id5]
                tri_indices[tri_start + 6] = [id4, id5, id6]
                tri_indices[tri_start + 7] = [id3, id6, id7]
                tri_indices[tri_start + 8] = [id4, id5, id6]
                tri_indices[tri_start + 9] = [id5, id6, id7]
                tri_indices[tri_start + 10] = [id1, id3, id7]
                tri_indices[tri_start + 11] = [id1, id5, id7]

    return points, tet_indices, tri_indices


# FIXME: still debugging
def generate_mesh_new(len, grid_dx=0.1):
    num_grid = int(len // grid_dx)
    points = np.zeros(((num_grid + 1) ** 3, 3), dtype=float)
    for i in range(num_grid + 1):
        for j in range(num_grid + 1):
            for k in range(num_grid + 1):
                points[i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k] = [i * grid_dx, j * grid_dx, k * grid_dx]

    cube = np.zeros((num_grid, num_grid, num_grid, 8), dtype=int)
    tet_indices = np.zeros((num_grid, num_grid, num_grid, 5, 4), dtype=int)
    for k in range(num_grid):
        for j in range(num_grid):
            for i in range(num_grid):
                cube[i, j, k, 0] = i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k
                cube[i, j, k, 1] = i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k + 1
                cube[i, j, k, 2] = i * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k
                cube[i, j, k, 3] = i * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k + 1
                cube[i, j, k, 4] = (i + 1) * (num_grid + 1) ** 2 + j * (num_grid + 1) + k
                cube[i, j, k, 5] = (i + 1) * (num_grid + 1) ** 2 + j * (num_grid + 1) + k + 1
                cube[i, j, k, 6] = (i + 1) * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k
                cube[i, j, k, 7] = (i + 1) * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k + 1

                v = cube[i, j, k, :]
                indx = [
                    [v[0], v[3], v[5], v[6]],
                    [v[0], v[1], v[3], v[5]],
                    [v[0], v[2], v[3], v[6]],
                    [v[0], v[4], v[5], v[6]],
                    [v[3], v[5], v[6], v[7]],
                ]
                tet_indices[i, j, k] = indx
    tet_indices = tet_indices.reshape(-1, 4)
    return points, tet_indices


def write_tet(filename, points, tet_indices):
    import meshio

    cells = [
        ("tetra", tet_indices),
    ]
    mesh = meshio.Mesh(
        points,
        cells,
    )
    mesh.write(filename)
    return mesh


# original size: 0.1 and 0.5
# fine_points, fine_tet_indices, fine_tri_indices = generate_mesh(2.0, 0.1)
# write_tet("data/model/cube/fine_new.node", fine_points, fine_tet_indices)
# coarse_points, coarse_tet_indices, coarse_tri_indices = generate_mesh(2.0, 0.5)
# write_tet("data/model/cube/coarse_new.node", coarse_points, coarse_tet_indices)

coarse_points, coarse_tet_indices, coarse_tri_indices = generate_cube_mesh(1.0, 1.0)
write_tet("data/model/cube/minicube.node", coarse_points, coarse_tet_indices)