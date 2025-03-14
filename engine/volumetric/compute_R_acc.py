import os
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import scipy
import tqdm
import argparse
import taichi as ti
from taichi.math import vec3,ivec4
from time import perf_counter
from pathlib import Path


# ti.init(default_fp=ti.f64)


@ti.func
def is_in_tet_func(p, p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = ti.math.inverse(A) @ b
    return ((x[0] >= 0 and x[1] >= 0 and x[2] >= 0) and x[0] + x[1] + x[2] <= 1), x


@ti.kernel
def compute_barycentric_kernel(
    p_pos: ti.template(),
    cage_vert_pos: ti.template(),
    cage_indx: ti.template(),
    which_cage: ti.template(),
    bary_coord: ti.template()
):
    n_p = p_pos.shape[0]
    n_cage = cage_indx.shape[0]
    cnt = 0
    for i in range(n_p):
        p = p_pos[i]
        flag = False
        for t in range(n_cage):
            a, b, c, d = cage_indx[t]
            p0, p1, p2, p3 = cage_vert_pos[a], cage_vert_pos[b], cage_vert_pos[c], cage_vert_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
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
                for idx in ti.static(range(4)):
                    dis = (p_tet[idx] - p).norm()
                    if dis < min_dis:
                        min_dis = dis
                        min_idx = t
            a, b, c, d = cage_indx[min_idx]
            p0, p1, p2, p3 = cage_vert_pos[a], cage_vert_pos[b], cage_vert_pos[c], cage_vert_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
            which_cage[i] = min_idx
            bary_coord[i] = x
    print(f"Totally {cnt} des verts not found cage, use the nearest tet instead")


# for each point in point set 1, find the nearest vert in point set 2 
@ti.kernel
def compute_nearest_point_kernel(
    pos1: ti.types.ndarray(dtype=vec3),
    pos2: ti.types.ndarray(dtype=vec3),
    nearest_in_2: ti.types.ndarray(dtype=int),
):
    for i in range(pos1.shape[0]):
        min_dis = 1e10
        nearest_in_2[i] = -1
        for j in range(pos2.shape[0]):
            dis = (pos1[i] - pos2[j]).norm()
            if dis < min_dis:
                min_dis = dis
                nearest_in_2[i] = j
            


def compute_mapping(coarse_pos, coarse_tet_indices, fine_pos, fine_tet_indices):
    coarse_nv = coarse_pos.shape[0]
    fine_nv = fine_pos.shape[0]

    coarse_in_fine_tet_indx = ti.field(dtype=ti.i32, shape=coarse_nv)
    coarse_in_fine_tet_coord = ti.Vector.field(3, dtype=ti.f32, shape=coarse_nv)
    fine_in_coarse_tet_indx = ti.field(dtype=ti.i32, shape=fine_nv)
    fine_in_coarse_tet_coord = ti.Vector.field(3, dtype=ti.f32, shape=fine_nv)

    fine_in_coarse_tet_indx.fill(-1)
    coarse_in_fine_tet_indx.fill(-1)

    cpos = ti.Vector.field(3, dtype=ti.f32, shape=coarse_nv)
    fpos = ti.Vector.field(3, dtype=ti.f32, shape=fine_nv)
    ctet = ti.Vector.field(4, dtype=ti.i32, shape=coarse_tet_indices.shape[0])
    ftet = ti.Vector.field(4, dtype=ti.i32, shape=fine_tet_indices.shape[0])
    cpos.from_numpy(coarse_pos)
    fpos.from_numpy(fine_pos)
    ctet.from_numpy(coarse_tet_indices)
    ftet.from_numpy(fine_tet_indices)

    print(">> Computing fine vert in which coarse cage...")
    compute_barycentric_kernel(
        fpos, cpos, ctet, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord
    )

    print(">> Computing coarse vert in which fine cage...")
    compute_barycentric_kernel(
        cpos, fpos, ftet, coarse_in_fine_tet_indx, coarse_in_fine_tet_coord
    )

    return coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord


def compute_mapping_v2(coarse_pos, coarse_tet_indices, fine_pos,):
    """
    计算从fine到coarse的映射，以及从coarse到fine的映射
    Args:
        coarse_pos: 粗网格顶点坐标，shape=(NV, 3)
        coarse_tet_indices: 粗网格单元索引，shape=(NT, 4)
        fine_pos: 细网格顶点坐标，shape=(NV, 3)
    Returns:    
        coarse2fine_nearest_vert: 每个粗网格顶点对应的最近细网格顶点索引，shape=(NV,)
        fine_in_coarse_tet_indx: 每个细网格顶点所在的粗网格单元索引，shape=(NV,)
        fine_in_coarse_tet_coord: 每个细网格顶点在所在粗网格单元中的重心坐标，shape=(NV, 3)
    """
    coarse_nv = coarse_pos.shape[0]
    fine_nv = fine_pos.shape[0]

    coarse2fine_nearest_vert = np.empty(coarse_nv, dtype=np.int32)
    fine_in_coarse_tet_indx = np.empty(fine_nv, dtype=np.int32)
    fine_in_coarse_tet_coord = np.zeros((fine_nv, 3), dtype=np.float64)

    coarse2fine_nearest_vert.fill(-1)
    fine_in_coarse_tet_indx.fill(-1)

    print(">> Computing fine vert in which coarse cage...")
    compute_barycentric_kernel(
        fine_pos, coarse_pos, coarse_tet_indices, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord
    )

    compute_nearest_point_kernel(coarse_pos, fine_pos, coarse2fine_nearest_vert)

    return coarse2fine_nearest_vert, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord



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


def read_tetgen_noface(filename):
    """
    读取tetgen生成的网格文件，返回顶点坐标、单元索引、面索引

    Args:
        filename: 网格文件名，不包含后缀名

    Returns:
        pos: 顶点坐标，shape=(NV, 3)
        tet_indices: 单元索引，shape=(NT, 4)
    """
    import numpy as np

    ele_file_name = filename + ".ele"
    node_file_name = filename + ".node"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(ele_file_name, "r") as f:
        lines = f.readlines()
        NT = int(lines[0].split()[0])
        tet_indices = np.zeros((NT, 4), dtype=np.int32)
        for i in range(NT):
            tet_indices[i] = np.array(lines[i + 1].split()[1:], dtype=np.int32)
    return pos, tet_indices


def build_cascadeCage_P(fine_model_path, coarse_model_path, suffix=""):
    start_time = perf_counter()

    print(f">> Reading mesh at {fine_model_path} and {coarse_model_path}...")
    coarse_pos, coarse_tet_indices = read_tetgen_noface(coarse_model_path)
    fine_pos, fine_tet_indices = read_tetgen_noface(fine_model_path)

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
    P = compute_P(n, m, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord, coarse_tet_indices)

    print(R@P-scipy.sparse.identity(m))


    mmwrite(str(Path(fine_model_path).parent) + "R" + suffix + ".mtx", R)
    mmwrite(str(Path(fine_model_path).parent) + "P" + suffix + ".mtx", P)

    end_time = perf_counter()
    print(f">> Total time: {end_time - start_time:.2f}s")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_model_path", type=str, default="data/model/cube/fine")
    parser.add_argument("--coarse_model_path", type=str, default="data/model/cube/coarse")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    build_cascadeCage_P(args.fine_model_path, args.coarse_model_path)