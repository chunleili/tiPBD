import os
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import tqdm
import argparse

sys.path.append(os.getcwd())
from engine.mesh_io import read_tetgen


def is_in_tet(p, p0, p1, p2, p3):
    A = np.array([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = np.linalg.solve(A, b)
    return (all(x >= 0) and sum(x) <= 1), x


def compute(p_pos, cage_vert_pos, cage_indx, which_cage, bary_coord):
    n_p = p_pos.shape[0]
    n_cage = cage_indx.shape[0]
    pbar = tqdm.tqdm(total=n_p)
    cnt = 0
    for i in range(n_p):
        pbar.update(1)
        p = p_pos[i]
        for t in range(n_cage):
            a, b, c, d = cage_indx[t]
            p0, p1, p2, p3 = cage_vert_pos[a], cage_vert_pos[b], cage_vert_pos[c], cage_vert_pos[d]
            flag, x = is_in_tet(p, p0, p1, p2, p3)
            if flag:
                which_cage[i] = t
                bary_coord[i] = x
                break
        # if des pos not in all tets, find the nearest tet
        if not flag or which_cage[i] < 0:
            cnt += 1
            # print(f"des vert {i}({p_pos[i]}) not in all tets, find the nearest tet, cnt ={cnt}")
            min_dis = 1e10
            min_idx = -1
            for t in range(n_cage):
                a, b, c, d = cage_indx[t]
                p_tet = [cage_vert_pos[a], cage_vert_pos[b], cage_vert_pos[c], cage_vert_pos[d]]
                for idx in range(4):
                    dis = np.linalg.norm(p_tet[idx] - p)
                    if dis < min_dis:
                        min_dis = dis
                        min_idx = t
            a, b, c, d = cage_indx[min_idx]
            p0, p1, p2, p3 = cage_vert_pos[a], cage_vert_pos[b], cage_vert_pos[c], cage_vert_pos[d]
            flag, x = is_in_tet(p, p0, p1, p2, p3)
            which_cage[i] = min_idx
            bary_coord[i] = x
    pbar.close()
    print(f"Totally {cnt} des verts not found cage, use the nearest tet instead")


def compute_mapping(coarse_pos, coarse_tet_indices, fine_pos, fine_tet_indices):
    coarse_nv = coarse_pos.shape[0]
    fine_nv = fine_pos.shape[0]

    coarse_in_fine_tet_indx = np.empty(coarse_nv, dtype=np.int32)
    coarse_in_fine_tet_coord = np.zeros((coarse_nv, 3), dtype=np.float64)
    fine_in_coarse_tet_indx = np.empty(fine_nv, dtype=np.int32)
    fine_in_coarse_tet_coord = np.zeros((fine_nv, 3), dtype=np.float64)

    fine_in_coarse_tet_indx.fill(-1)
    coarse_in_fine_tet_indx.fill(-1)

    print(">> Computing fine vert in which coarse cage...")
    compute(fine_pos, coarse_pos, coarse_tet_indices, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord)

    print(">> Computing coarse vert in which fine cage...")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cube")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    model_path = "data/model/cube/"
    fine_model_path = model_path + "fine"
    coarse_model_path = model_path + "coarse"
    if args.model == "bunny":
        model_path = "data/model/bunny1k2k/"
        fine_model_path = model_path + "bunny2k"
        coarse_model_path = model_path + "bunny1k"

    print(f">> Reading mesh at {model_path}...")
    coarse_pos, coarse_tet_indices, coarse_face_indices = read_tetgen(coarse_model_path)
    fine_pos, fine_tet_indices, fine_face_indices = read_tetgen(fine_model_path)

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
    mmwrite(model_path + "R" + args.suffix + ".mtx", R)
    P = compute_P(n, m, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord, coarse_tet_indices)
    mmwrite(model_path + "P" + args.suffix + ".mtx", P)
