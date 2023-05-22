import os
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import tqdm

sys.path.append(os.getcwd())
from engine.mesh_io import read_tetgen


def is_in_tet(p, p0, p1, p2, p3):
    A = np.array([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = np.linalg.solve(A, b)
    return (all(x >= 0) and sum(x) <= 1), x


def compute(des_pos, src_pos, src_tet_indices, des_in_src_tet_indx, des_in_src_tet_coord):
    des_nv = des_pos.shape[0]
    src_nt = src_tet_indices.shape[0]
    pbar = tqdm.tqdm(total=des_nv)
    for i in range(des_nv):
        pbar.update(1)
        p = des_pos[i]
        for t in range(src_nt):
            a, b, c, d = src_tet_indices[t]
            p0, p1, p2, p3 = src_pos[a], src_pos[b], src_pos[c], src_pos[d]
            flag, x = is_in_tet(p, p0, p1, p2, p3)
            if flag:
                des_in_src_tet_indx[i] = t
                des_in_src_tet_coord[i] = x
                break
        # if des pos not in all tets, find the nearest tet
        if not flag or des_in_src_tet_indx[i] < 0:
            min_dis = 1e10
            min_idx = -1
            for t in range(src_nt):
                a, b, c, d = src_tet_indices[t]
                p_tet = [src_pos[a], src_pos[b], src_pos[c], src_pos[d]]
                for idx in range(4):
                    dis = np.linalg.norm(p_tet[idx] - p)
                    if dis < min_dis:
                        min_dis = dis
                        min_idx = t
            a, b, c, d = src_tet_indices[min_idx]
            p0, p1, p2, p3 = src_pos[a], src_pos[b], src_pos[c], src_pos[d]
            flag, x = is_in_tet(p, p0, p1, p2, p3)
            des_in_src_tet_indx[i] = min_idx
            des_in_src_tet_coord[i] = x
    pbar.close()


def compute_mapping(coarse_pos, coarse_tet_indices, fine_pos, fine_tet_indices):
    coarse_nv = coarse_pos.shape[0]
    fine_nv = fine_pos.shape[0]

    coarse_in_fine_tet_indx = np.empty(coarse_nv, dtype=np.int32)
    coarse_in_fine_tet_coord = np.zeros((coarse_nv, 3), dtype=np.float64)
    fine_in_coarse_tet_indx = np.empty(fine_nv, dtype=np.int32)
    fine_in_coarse_tet_coord = np.zeros((fine_nv, 3), dtype=np.float64)

    fine_in_coarse_tet_indx.fill(-1)
    coarse_in_fine_tet_indx.fill(-1)

    compute(fine_pos, coarse_pos, coarse_tet_indices, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord)
    compute(coarse_pos, fine_pos, fine_tet_indices, coarse_in_fine_tet_indx, coarse_in_fine_tet_coord)

    return coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord


def compute_R(n, m, coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_tet_indices):
    """
    Compute restriction operator R:
            x_c = R @ x_f, x_c is coarse vertex positions, x_f is fine vertex positions
    Parameters:
    n: number of fine vertices
    m: number of coarse vertices
    Output:
        R_coo: restriction operator in coo format
    """
    row = np.zeros(4 * m, dtype=np.int32)
    col = np.zeros(4 * m, dtype=np.int32)
    val = np.zeros(4 * m, dtype=np.float64)
    for i in range(m):
        row[4 * i : 4 * i + 4] = [i, i, i, i]
        fine_idx = coarse_in_fine_tet_indx[i]
        a, b, c, d = fine_tet_indices[fine_idx]
        u, v, w = coarse_in_fine_tet_coord[i]
        col[4 * i : 4 * i + 4] = [a, b, c, d]
        val[4 * i : 4 * i + 4] = [1 - u - v - w, u, v, w]
    R_coo = coo_matrix((val, (row, col)), shape=(m, n))
    return R_coo


def compute_P(n, m, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord, coarse_tet_indices):
    """
    Compute prolongation operator P
    n: number of fine vertices
    m: number of coarse vertices
    """
    row = np.zeros(4 * n, dtype=np.int32)
    col = np.zeros(4 * n, dtype=np.int32)
    val = np.zeros(4 * n, dtype=np.float64)
    for i in range(n):
        row[4 * i : 4 * i + 4] = [i, i, i, i]
        coarse_idx = fine_in_coarse_tet_indx[i]
        a, b, c, d = coarse_tet_indices[coarse_idx]
        col[4 * i : 4 * i + 4] = [a, b, c, d]
        u, v, w = fine_in_coarse_tet_coord[i]
        val[4 * i : 4 * i + 4] = [1 - u - v - w, u, v, w]
    P_coo = coo_matrix((val, (row, col)), shape=(n, m))
    return P_coo


if __name__ == "__main__":
    path = "data/model/cube/"

    fine_mesh = path + "fine"
    coarse_mesh = path + "coarse"

    print(">> Reading mesh...")
    coarse_pos, coarse_tet_indices, coarse_face_indices = read_tetgen(coarse_mesh)
    fine_pos, fine_tet_indices, fine_face_indices = read_tetgen(fine_mesh)

    print(">> Start to compute coarse and fine mapping...")
    (
        coarse_in_fine_tet_indx,
        coarse_in_fine_tet_coord,
        fine_in_coarse_tet_indx,
        fine_in_coarse_tet_coord,
    ) = compute_mapping(coarse_pos, coarse_tet_indices, fine_pos, fine_tet_indices)

    print(">> Start to compute R and P...")
    n = fine_pos.shape[0]
    m = coarse_pos.shape[0]
    R = compute_R(n, m, coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_tet_indices)
    mmwrite(path + "R1.mtx", R)
    P = compute_P(n, m, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord, coarse_tet_indices)
    mmwrite(path + "P1.mtx", P)
