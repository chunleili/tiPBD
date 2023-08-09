'''
和前一版本相比，更改控制所有在四面体内的细网格。（按质心算位置）
'''
import os
import sys
import numpy as np
import scipy
import argparse
import taichi as ti
from taichi.math import vec3, ivec4
from time import perf_counter
import meshio
import pathlib
from time import time


sys.path.append(os.getcwd())

ti.init(default_fp=ti.f64)


@ti.func
def is_in_tet_func(p, p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = ti.math.inverse(A) @ b
    return ((x[0] >= 0 and x[1] >= 0 and x[2] >= 0) and x[0] + x[1] + x[2] <= 1), x



def compute_p_to_tet(fine_tet_indices, p_to_tet):
    for t in range(fine_tet_indices.shape[0]):
        ia, ib, ic, id = fine_tet_indices[t]
        p_to_tet[ia] = t
        p_to_tet[ib] = t
        p_to_tet[ic] = t
        p_to_tet[id] = t


@ti.kernel
def normalize_R_by_row(R: ti.types.ndarray()):
    for i in range(R.shape[0]):
        sum = 0.0
        for j in range(R.shape[1]):
            sum += R[i, j]
        for j in range(R.shape[1]):
            R[i, j] /= sum


def read_tet(filename):
    mesh = meshio.read(filename)
    pos = mesh.points
    tet_indices = mesh.cells_dict["tetra"]
    return pos, tet_indices


@ti.func
def tet_centroid_func(tet_indices, pos, t):
    a, b, c, d = tet_indices[t]
    p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
    p = (p0 + p1 + p2 + p3) / 4
    return p

@ti.kernel
def compute_all_centroid(pos:ti.template(),tet_indices:ti.template(), res:ti.template()):
    for t in range(tet_indices.shape[0]):
        a, b, c, d = tet_indices[t]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        p = (p0 + p1 + p2 + p3) / 4
        res[t] = p



@ti.kernel
def compute_R_kernel_new(
    fine_pos: ti.template(),
    fine_tet_indices: ti.template(),
    fine_centroid: ti.template(),
    coarse_pos: ti.template(),
    coarse_tet_indices: ti.template(),
    coarse_centroid: ti.template(),
    R: ti.types.sparse_matrix_builder(),
):
    for i in fine_centroid:
        p = fine_centroid[i]
        flag = False
        for tc in range(coarse_tet_indices.shape[0]):
            a, b, c, d = coarse_tet_indices[tc]
            p0, p1, p2, p3 = coarse_pos[a], coarse_pos[b], coarse_pos[c], coarse_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
            if flag:
                R[tc, i] += 1
                break
        if not flag:
            print("Warning: fine tet centroid {i} not in any coarse tet")





if __name__ == "__main__":
    start_time = perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse_model_path", type=str, default="data/model/cube/coarse_new.node")
    parser.add_argument("--fine_model_path", type=str, default="data/model/cube/fine_new.node")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    print(f">> Reading mesh...")
    coarse_pos_np, coarse_tet_indices_np = read_tet(args.coarse_model_path)
    fine_pos_np, fine_tet_indices_np = read_tet(args.fine_model_path)

    fine_pos = ti.Vector.field(3, dtype=ti.f64, shape=fine_pos_np.shape[0])
    fine_tet_indices = ti.Vector.field(4, dtype=ti.i32, shape=fine_tet_indices_np.shape[0])
    fine_centroid = ti.Vector.field(3,dtype=ti.f64, shape=fine_tet_indices_np.shape[0])
    coarse_pos = ti.Vector.field(3, dtype=ti.f64, shape=coarse_pos_np.shape[0])
    coarse_tet_indices = ti.Vector.field(4, dtype=ti.i32, shape=coarse_tet_indices_np.shape[0])
    coarse_centroid = ti.Vector.field(3,dtype=ti.f64, shape=coarse_tet_indices_np.shape[0])


    fine_pos.from_numpy(fine_pos_np)
    coarse_pos.from_numpy(coarse_pos_np)
    fine_tet_indices.from_numpy(fine_tet_indices_np)
    coarse_tet_indices.from_numpy(coarse_tet_indices_np)


    # 计算所有四面体的质心
    print(">>Computing all tet centroid...")
    compute_all_centroid(fine_pos, fine_tet_indices, fine_centroid)
    compute_all_centroid(coarse_pos, coarse_tet_indices, coarse_centroid)


    # 计算R 和 P
    print(">>Computing P and R...")
    t = time()
    M, N = coarse_tet_indices.shape[0], fine_tet_indices.shape[0]
    R_builder = ti.linalg.SparseMatrixBuilder(M, N, max_num_triplets=40 * M)
    compute_R_kernel_new(fine_pos, fine_tet_indices, fine_centroid, coarse_pos, coarse_tet_indices, coarse_centroid, R_builder)
    R = R_builder.build()
    P = R.transpose()
    print(f"Computing P and R done, time = {time() - t}")
    print(f"writing P and R...")
    R.mmwrite("R.mtx")
    P.mmwrite("P.mtx")
    print(f"writing P and R done")