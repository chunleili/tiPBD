#source: https://github.com/FantasyVR/amg_pbd/blob/46b8907275839d86d8cff690c00a01520904f969/arap_xpbd/arap_3d/generate_coarse_fine_mesh.py

import numpy as np


def generate_mesh(len, grid_dx=0.1):
    num_grid = int(len // grid_dx)
    points = np.zeros(((num_grid + 1)**3, 3), dtype=float)
    for i in range(num_grid + 1):
        for j in range(num_grid + 1):
            for k in range(num_grid + 1):
                points[i * (num_grid + 1)**2 + j * (num_grid + 1) +
                       k] = [i * grid_dx, j * grid_dx, k * grid_dx]
    tet_indices = np.zeros(((num_grid)**3 * 5, 4), dtype=int)
    tri_indices = np.zeros(((num_grid)**3 * 12, 3), dtype=int)
    for i in range(num_grid):
        for j in range(num_grid):
            for k in range(num_grid):
                id0 = i * (num_grid + 1)**2 + j * (num_grid + 1) + k
                id1 = i * (num_grid + 1)**2 + j * (num_grid + 1) + k + 1
                id2 = i * (num_grid + 1)**2 + (j + 1) * (num_grid + 1) + k
                id3 = i * (num_grid + 1)**2 + (j + 1) * (num_grid + 1) + k + 1
                id4 = (i + 1) * (num_grid + 1)**2 + j * (num_grid + 1) + k
                id5 = (i + 1) * (num_grid + 1)**2 + j * (num_grid + 1) + k + 1
                id6 = (i + 1) * (num_grid + 1)**2 + (j + 1) * (num_grid +
                                                               1) + k
                id7 = (i + 1) * (num_grid + 1)**2 + (j + 1) * (num_grid +
                                                               1) + k + 1
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


def save_mesh(points, tet_indices, tri_indices, filename):
    node_mesh = filename + '.node'
    ele_mesh = filename + '.ele'
    face_mesh = filename + '.face'
    with open(node_mesh, 'w') as f:
        f.write(f"{points.shape[0]} 3 0 0\n")
        for i in range(points.shape[0]):
            f.write(f"{i} {points[i, 0]} {points[i, 1]} {points[i, 2]}\n")
    with open(ele_mesh, 'w') as f:
        f.write(f"{tet_indices.shape[0]} 4 0\n")
        for i in range(tet_indices.shape[0]):
            f.write(
                f"{i} {tet_indices[i, 0]} {tet_indices[i, 1]} {tet_indices[i, 2]} {tet_indices[i, 3]}\n"
            )
    with open(face_mesh, 'w') as f:
        f.write(f"{tri_indices.shape[0]} 3 0\n")
        for i in range(tri_indices.shape[0]):
            f.write(
                f"{i} {tri_indices[i, 0]} {tri_indices[i, 1]} {tri_indices[i, 2]} -1\n"
            )


def generate_random_points(num_points, len):
    points = np.random.rand(num_points, 3) * len
    return points

# original size: 0.1 and 0.5
fine_dx, coarse_dx = 0.05, 0.25
fine_points, fine_tet_indices, fine_tri_indices = generate_mesh(2.0, fine_dx)
save_mesh(fine_points, fine_tet_indices, fine_tri_indices, 'data/model/cube_large/fine')
# fine_init_random_points = generate_random_points(fine_points.shape[0], 2.0)
# np.savetxt('cube/fine_init_random_points.txt', fine_init_random_points)
coarse_points, coarse_tet_indices, coarse_tri_indices = generate_mesh(2.0, coarse_dx)
save_mesh(coarse_points, coarse_tet_indices, coarse_tri_indices, 'data/model/cube_large/coarse')