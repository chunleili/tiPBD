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

sys.path.append(os.getcwd())

ti.init(default_fp=ti.f64)


@ti.func
def is_in_tet_func(p, p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = ti.math.inverse(A) @ b
    return ((x[0] >= 0 and x[1] >= 0 and x[2] >= 0) and x[0] + x[1] + x[2] <= 1), x


@ti.kernel
def compute_R(
    fine_pos: ti.types.ndarray(dtype=vec3),
    p_to_tet: ti.types.ndarray(),
    coarse_pos: ti.types.ndarray(dtype=vec3),
    coarse_tet_indices: ti.types.ndarray(dtype=ivec4),
    R: ti.types.ndarray(),
):
    for i in fine_pos:
        p = fine_pos[i]
        flag = False
        tf = p_to_tet[i]
        for tc in range(coarse_tet_indices.shape[0]):
            a, b, c, d = coarse_tet_indices[tc]
            p0, p1, p2, p3 = coarse_pos[a], coarse_pos[b], coarse_pos[c], coarse_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
            if flag:
                R[tc, tf] = 1
                break
        if not flag:
            print(f"WARNING: point {i} not in any tet")
            min_dist = 1e10
            min_indx = -1
            for ic in coarse_pos:
                dist = (p - coarse_pos[ic]).norm()
                if dist < min_dist:
                    min_dist = dist
                    min_indx = ic
            tc = p_to_tet[min_indx]
            print(
                f"fine point {i}({p}) closest to coarse point {min_indx}({coarse_pos[min_indx]}), which is in tet {tc}, min_dist = {min_dist}"
            )
            R[tc, tf] = 1


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


if __name__ == "__main__":
    start_time = perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse_model_path", type=str, default="data/model/cube/coarse.node")
    parser.add_argument("--fine_model_path", type=str, default="data/model/cube/fine.node")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    print(f">> Reading mesh...")
    coarse_pos, coarse_tet_indices = read_tet(args.coarse_model_path)
    fine_pos, fine_tet_indices = read_tet(args.fine_model_path)

    p_to_tet = np.empty(shape=(fine_pos.shape[0]), dtype=np.int32)
    p_to_tet.fill(-1)
    compute_p_to_tet(fine_tet_indices, p_to_tet)

    R = np.zeros((coarse_tet_indices.shape[0], fine_tet_indices.shape[0]))
    print("R shape: ", R.shape)

    print("Computing R(fine ele in which coarse ele)...")
    fine_pos = np.ascontiguousarray(fine_pos)
    coarse_pos = np.ascontiguousarray(coarse_pos)
    compute_R(fine_pos, p_to_tet, coarse_pos, coarse_tet_indices, R)

    # print("Normalizing R by row...")
    # normalize_R_by_row(R)

    print("Computing P by transpose(R)...")
    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()

    scipy.io.mmwrite(args.output_path + "R" + args.output_suffix + ".mtx", R)
    scipy.io.mmwrite(args.output_path + "P" + args.output_suffix + ".mtx", P)
    end_time = perf_counter()
    print(f">> Total time: {end_time - start_time:.2f}s")
