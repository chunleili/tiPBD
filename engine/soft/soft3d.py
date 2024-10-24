import taichi as ti
from taichi.lang.ops import sqrt
import numpy as np
import logging
from logging import info, warning
import scipy
import scipy.sparse as sparse
import sys, os, argparse
import time
from time import perf_counter
from pathlib import Path
import meshio
from collections import namedtuple
import json
from functools import singledispatch
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial
import pyamg
import ctypes
import numpy.ctypeslib as ctl
import datetime
import tqdm

prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("-maxiter", type=int, default=50)
parser.add_argument("-omega", type=float, default=0.1)
parser.add_argument("-mu", type=float, default=1e6)
parser.add_argument("-delta_t", type=float, default=1e-3)
parser.add_argument("-damping_coeff", type=float, default=1.0)
parser.add_argument("-gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("-total_mass", type=float, default=16000.0)
parser.add_argument("-solver_type", type=str, default="AMG", choices=["XPBD",  "AMG", "AMGX"])
parser.add_argument("-model_path", type=str, default=f"data/model/bunny1k2k/coarse.node")
# "data/model/cube/minicube.node"
# "data/model/bunny1k2k/coarse.node"
# "data/model/bunny_small/bunny_small.node"
# "data/model/bunnyBig/bunnyBig.node"
# "data/model/bunny85w/bunny85w.node"
parser.add_argument("-end_frame", type=int, default=10)
parser.add_argument("-out_dir", type=str, default="result/latest/")
parser.add_argument("-export_matrix", type=int, default=False)
parser.add_argument("-export_matrix_binary", type=int, default=True)
parser.add_argument("-auto_another_outdir", type=int, default=False)
parser.add_argument("-use_cuda", type=int, default=True)
parser.add_argument("-cuda_dir", type=str, default="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
parser.add_argument("-smoother_type", type=str, default="jacobi")
parser.add_argument("-build_P_method", type=str, default="UA")
parser.add_argument("-arch", type=str, default="cpu")
parser.add_argument("-setup_interval", type=int, default=20)
parser.add_argument("-maxiter_Axb", type=int, default=100)
parser.add_argument("-export_log", type=int, default=True)
parser.add_argument("-export_residual", type=int, default=False)
parser.add_argument("-restart_frame", type=int, default=-1)
parser.add_argument("-restart", type=int, default=False)
parser.add_argument("-use_cache", type=int, default=True)
parser.add_argument("-export_mesh", type=int, default=True)
parser.add_argument("-reinit", type=str, default="enlarge", choices=["", "random", "enlarge"])
parser.add_argument("-tol", type=float, default=1e-4)
parser.add_argument("-rtol", type=float, default=1e-9)
parser.add_argument("-tol_Axb", type=float, default=1e-5)
parser.add_argument("-large", action="store_true")
parser.add_argument("-samll", action="store_true")
parser.add_argument("-amgx_config", type=str, default="data/amgx_config/AMG_CONFIG_CG.json")
parser.add_argument("-smoother_niter", type=int, default=2)
parser.add_argument("-filter_P", type=str, default=None)
parser.add_argument("-scale_RAP", type=int, default=False)
parser.add_argument("-only_smoother", type=int, default=False)
parser.add_argument("-debug", type=int, default=False)

args = parser.parse_args()

out_dir = args.out_dir
use_graph_coloring = False
if args.smoother_type=="gauss_seidel":
    use_graph_coloring =True

if args.large:
    args.model_path = f"data/model/bunny85w/bunny85w.node"
if args.samll:
    args.model_path = f"data/model/bunny1k2k/coarse.node"

if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

t_export = 0.0
t_avg_iter = []

n_outer_all = []
ResidualData = namedtuple('residual', ['dual', 'ninner','t']) #residual for one outer iter

arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
arr2d_float = ctl.ndpointer(dtype=np.float32, ndim=2, flags='aligned, c_contiguous')
arr2d_int = ctl.ndpointer(dtype=np.int32, ndim=2, flags='aligned, c_contiguous')
arr3d_int = ctl.ndpointer(dtype=np.int32, ndim=3, flags='aligned, c_contiguous')
arr3d_int8 = ctl.ndpointer(dtype=np.int8, ndim=3, flags='aligned, c_contiguous')
arr3d_float = ctl.ndpointer(dtype=np.float32, ndim=3, flags='aligned, c_contiguous')
c_size_t = ctypes.c_size_t
c_float = ctypes.c_float
c_int = ctypes.c_int
argtypes_of_csr=[ctl.ndpointer(np.float32,flags='aligned, c_contiguous'),    # data
                ctl.ndpointer(np.int32,  flags='aligned, c_contiguous'),      # indices
                ctl.ndpointer(np.int32,  flags='aligned, c_contiguous'),      # indptr
                ctypes.c_int, ctypes.c_int, ctypes.c_int           # rows, cols, nnz
                ]

def init_extlib_argtypes():
    global extlib

    # # # DEBUG only
    if args.debug:
        os.chdir(prj_path+'/cpp/mgcg_cuda')
        os.system("cmake --build build --config Debug")
        os.chdir(prj_path)

    os.add_dll_directory(args.cuda_dir)
    extlib = ctl.load_library("fastmg.dll", prj_path+'/cpp/mgcg_cuda/lib')

    extlib.fastmg_set_data.argtypes = [arr_float, c_size_t, arr_float, c_size_t, c_float, c_size_t]
    extlib.fastmg_get_data.argtypes = [arr_float]*2
    extlib.fastmg_get_data.restype = c_size_t
    extlib.argtypes = [ctypes.c_float, ctypes.c_size_t]
    extlib.fastmg_RAP.argtypes = [ctypes.c_size_t]
    extlib.fastmg_set_A0.argtypes = argtypes_of_csr
    extlib.fastmg_set_P.argtypes = [ctypes.c_size_t] + argtypes_of_csr
    extlib.fastmg_set_smoother_niter.argtypes = [ctypes.c_size_t]
    extlib.fastmg_update_A0.argtypes = [arr_float]

    extlib.fastFillSoft_set_data.argtypes = [arr2d_int, c_int, arr_float, c_int, arr2d_float, arr_float]
    extlib.fastFillSoft_fetch_A_data.argtypes = [arr_float]
    extlib.fastFillSoft_run.argtypes = [arr2d_float, arr3d_float]

    extlib.fastmg_new()
    extlib.fastFillSoft_new()
    if args.scale_RAP:
        extlib.fastmg_scale_RAP.argtypes = [c_float, c_int]

if args.use_cuda:
    init_extlib_argtypes()

class Meta:
    def __init__(self) -> None:
        # control parameters
        self.args = args
        self.frame = 0
        self.ite = 0

        # physical parameters
        self.omega = self.args.omega  # SOR factor, default 0.1
        self.mu = self.args.mu  # Lame's second parameter, default 1e6
        self.inv_mu = 1.0 / self.mu
        self.delta_t = self.args.delta_t  # time step size, default 3e-3
        self.inv_h2 = 1.0 / self.delta_t / self.delta_t
        self.gravity = ti.Vector(self.args.gravity)  # gravity, default (0, 0, 0)
        self.damping_coeff = self.args.damping_coeff  # damping coefficient, default 1.0
        self.total_mass = self.args.total_mass  # total mass, default 16000.0
        # self.mass_density = 2000.0


meta = Meta()


def clean_result_dir(folder_path):
    import glob
    print(f"clean {folder_path}...")
    to_remove = []
    for name in [
        '*.txt',
        '*.obj',
        '*.png',
        '*.ply',
        '*.npz',
        '*.mtx',
        '*.log',
    ]:
        files = glob.glob(os.path.join(folder_path, name))
        to_remove += (files)
    print(f"removing {len(to_remove)} files")
    for file_path in to_remove:
        os.remove(file_path)
    print(f"clean {folder_path} done")


def write_mesh(filename, pos, tri, format="ply"):
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )

    if format == "ply":
        mesh.write(filename + ".ply", binary=True)
    elif format == "obj":
        mesh.write(filename + ".obj")
    else:
        raise ValueError("Unknown format")
    return mesh


def read_tetgen(filename):
    """
    读取tetgen生成的网格文件，返回顶点坐标、单元索引、面索引

    Args:
        filename: 网格文件名，不包含后缀名

    Returns:
        pos: 顶点坐标，shape=(NV, 3)
        tet_indices: 单元索引，shape=(NT, 4)
        face_indices: 面索引，shape=(NF, 3)
    """
    ele_file_name = filename + ".ele"
    node_file_name = filename + ".node"
    face_file_name = filename + ".face"

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

    with open(face_file_name, "r") as f:
        lines = f.readlines()
        NF = int(lines[0].split()[0])
        face_indices = np.zeros((NF, 3), dtype=np.int32)
        for i in range(NF):
            face_indices[i] = np.array(lines[i + 1].split()[1:-1], dtype=np.int32)

    return pos, tet_indices, face_indices


def read_tet(filename, build_face_flag=False):
    mesh = meshio.read(filename)
    pos = mesh.points
    tet_indices = mesh.cells_dict["tetra"]
    if build_face_flag:
        face_indices = build_face_indices(tet_indices)
        return pos, tet_indices, face_indices
    else:
        return pos, tet_indices


def build_face_indices(tet_indices):
    face_indices = np.empty((tet_indices.shape[0] * 4, 3), dtype=np.int32)
    for t in range(tet_indices.shape[0]):
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in range(4):  # 4 faces
            for j in range(3):  # 3 vertices
                face_indices[t * 4 + i][j] = tet_indices[t][ind[i][j]]
    return face_indices


# write .node file
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

# Usage:
# # original size: 0.1 and 0.5
# coarse_points, coarse_tet_indices, coarse_tri_indices = generate_cube_mesh(1.0, 1.0)
# write_tet("data/model/cube/minicube.node", coarse_points, coarse_tet_indices)
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
    write_tet("data/model/cube/coarse_new.node", points, tet_indices)
    return points, tet_indices, tri_indices


class SoftBody:
    def __init__(self, path):
        tic = time.perf_counter()
        self.model_pos, self.model_tet, self.model_tri = read_tet(path, build_face_flag=True)
        print(f"read_tet cost: {time.perf_counter() - tic:.4f}s")
        self.NV = len(self.model_pos)
        self.NT = len(self.model_tet)
        self.NF = len(self.model_tri)
        self.display_indices = ti.field(ti.i32, self.NF * 3)
        self.display_indices.from_numpy(self.model_tri.flatten())

        self.pos = ti.Vector.field(3, float, self.NV)
        self.pos_mid = ti.Vector.field(3, float, self.NV)
        self.predict_pos = ti.Vector.field(3, float, self.NV)
        self.old_pos = ti.Vector.field(3, float, self.NV)
        self.vel = ti.Vector.field(3, float, self.NV)  # velocity of particles
        self.mass = ti.field(float, self.NV)  # mass of particles
        self.inv_mass = ti.field(float, self.NV)  # inverse mass of particles
        self.tet_indices = ti.Vector.field(4, int, self.NT)
        self.B = ti.Matrix.field(3, 3, float, self.NT)  # D_m^{-1}
        self.lagrangian = ti.field(float, self.NT)  # lagrangian multipliers
        self.rest_volume = ti.field(float, self.NT)  # rest volume of each tet
        self.inv_V = ti.field(float, self.NT)  # inverse volume of each tet
        self.alpha_tilde = ti.field(float, self.NT)

        self.par_2_tet = ti.field(int, self.NV)
        self.gradC = ti.Vector.field(3, ti.f32, shape=(self.NT, 4))
        self.constraint = ti.field(ti.f32, shape=(self.NT))
        self.dpos = ti.Vector.field(3, ti.f32, shape=(self.NV))
        self.residual = ti.field(ti.f32, shape=self.NT)
        self.dual_residual = ti.field(ti.f32, shape=self.NT)
        self.dlambda = ti.field(ti.f32, shape=self.NT)
        self.tet_centroid = ti.Vector.field(3, ti.f32, shape=self.NT)
        self.potential_energy = ti.field(ti.f32, shape=())
        self.inertial_energy = ti.field(ti.f32, shape=())

        self.ele = self.tet_indices

        self.state = [
            self.pos,
        ]

        info(f"Creating instance done")

    def initialize(self):
        info(f"Initializing mesh")

        # read models
        self.model_pos = self.model_pos.astype(np.float32)
        self.model_tet = self.model_tet.astype(np.int32)
        self.pos.from_numpy(self.model_pos)
        self.tet_indices.from_numpy(self.model_tet)

        # init inv_mass rest volume alpha_tilde etc.
        init_physics(
            self.pos,
            self.old_pos,
            self.vel,
            self.tet_indices,
            self.B,
            self.rest_volume,
            self.inv_V,
            self.mass,
            self.inv_mass,
            self.alpha_tilde,
            self.par_2_tet,
        )

        # FIXME: no reinit will cause bug, why?
        # reinit pos
        if args.reinit == "random":
            # random init
            random_val = np.random.rand(self.pos.shape[0], 3)
            self.pos.from_numpy(random_val)
        elif args.reinit == "enlarge":
            # init by enlarge 1.5x
            self.pos.from_numpy(self.model_pos * 1.5)

        self.alpha_tilde_np = ist.alpha_tilde.to_numpy()


    def solve_constraints(self):
        solve_constraints_kernel(
            self.pos_mid,
            self.tet_indices,
            self.inv_mass,
            self.lagrangian,
            self.B,
            self.pos,
            self.alpha_tilde,
            self.constraint,
            self.residual,
            self.gradC,
            self.dlambda,
        )


# ---------------------------------------------------------------------------- #
#                                    kernels                                   #
# ---------------------------------------------------------------------------- #


@ti.kernel
def fill_gradC_triplets_kernel(
    ii:ti.types.ndarray(dtype=ti.i32),
    jj:ti.types.ndarray(dtype=ti.i32),
    vv:ti.types.ndarray(dtype=ti.f32),
    gradC: ti.template(),
    tet_indices: ti.template(),
):
    cnt=0
    ti.loop_config(serialize=True)
    for j in range(tet_indices.shape[0]):
        ind = tet_indices[j]
        for p in range(4):
            for d in range(3):
                pid = ind[p]
                ii[cnt],jj[cnt],vv[cnt] = j, 3 * pid + d, gradC[j, p][d]
                cnt+=1


@ti.kernel
def init_physics(
    pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    rest_volume: ti.template(),
    inv_V: ti.template(),
    mass: ti.template(),
    inv_mass: ti.template(),
    alpha_tilde: ti.template(),
    par_2_tet: ti.template(),
):
    # init pos, old_pos, vel
    for i in pos:
        old_pos[i] = pos[i]
        vel[i] = ti.Vector([0, 0, 0])

    # init B and rest_volume
    total_volume = 0.0
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        p0, p1, p2, p3 = pos[ia], pos[ib], pos[ic], pos[id]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        B[i] = D_m.inverse()

        rest_volume[i] = 1.0 / 6.0 * ti.abs(D_m.determinant())
        inv_V[i] = 1.0 / rest_volume[i]
        total_volume += rest_volume[i]

    # init mass
    # for i in tet_indices:
    #     ia, ib, ic, id = tet_indices[i]
    #     mass_density = meta.total_mass / total_volume
    #     tet_mass = mass_density * rest_volume[i]
    #     avg_mass = tet_mass / 4.0
    #     mass[ia] += avg_mass
    #     mass[ib] += avg_mass
    #     mass[ic] += avg_mass
    #     mass[id] += avg_mass
    for i in inv_mass:
        inv_mass[i] = 1.0 

    for i in alpha_tilde:
        alpha_tilde[i] = meta.inv_h2 * meta.inv_mu * inv_V[i]

    # init par_2_tet
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        par_2_tet[ia], par_2_tet[ib], par_2_tet[ic], par_2_tet[id] = i, i, i, i


@ti.kernel
def reset_lagrangian(lagrangian: ti.template()):
    for i in lagrangian:
        lagrangian[i] = 0.0


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix(
        [
            [x, 0, 0, y, 0, 0, z, 0, 0],
            [0, x, 0, 0, y, 0, 0, z, 0],
            [0, 0, x, 0, 0, y, 0, 0, z],
        ]
    )


@ti.func
def compute_gradient(U, S, V, B):
    sum_sigma = sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)

    # (dcdS00, dcdS11, dcdS22)
    dcdS = 1.0 / sum_sigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1, S[2, 2] - 1])
    # Compute (dFdx)^T
    dFdp1T = make_matrix(B[0, 0], B[0, 1], B[0, 2])
    dFdp2T = make_matrix(B[1, 0], B[1, 1], B[1, 2])
    dFdp3T = make_matrix(B[2, 0], B[2, 1], B[2, 2])
    # Compute (dsdF)
    u00, u01, u02 = U[0, 0], U[0, 1], U[0, 2]
    u10, u11, u12 = U[1, 0], U[1, 1], U[1, 2]
    u20, u21, u22 = U[2, 0], U[2, 1], U[2, 2]
    v00, v01, v02 = V[0, 0], V[0, 1], V[0, 2]
    v10, v11, v12 = V[1, 0], V[1, 1], V[1, 2]
    v20, v21, v22 = V[2, 0], V[2, 1], V[2, 2]
    dsdF00 = ti.Vector([u00 * v00, u01 * v01, u02 * v02])
    dsdF10 = ti.Vector([u10 * v00, u11 * v01, u12 * v02])
    dsdF20 = ti.Vector([u20 * v00, u21 * v01, u22 * v02])
    dsdF01 = ti.Vector([u00 * v10, u01 * v11, u02 * v12])
    dsdF11 = ti.Vector([u10 * v10, u11 * v11, u12 * v12])
    dsdF21 = ti.Vector([u20 * v10, u21 * v11, u22 * v12])
    dsdF02 = ti.Vector([u00 * v20, u01 * v21, u02 * v22])
    dsdF12 = ti.Vector([u10 * v20, u11 * v21, u12 * v22])
    dsdF22 = ti.Vector([u20 * v20, u21 * v21, u22 * v22])

    # Compute (dcdF)
    dcdF = ti.Vector(
        [
            dsdF00.dot(dcdS),
            dsdF10.dot(dcdS),
            dsdF20.dot(dcdS),
            dsdF01.dot(dcdS),
            dsdF11.dot(dcdS),
            dsdF21.dot(dcdS),
            dsdF02.dot(dcdS),
            dsdF12.dot(dcdS),
            dsdF22.dot(dcdS),
        ]
    )
    g1 = dFdp1T @ dcdF
    g2 = dFdp2T @ dcdF
    g3 = dFdp3T @ dcdF
    g0 = -g1 - g2 - g3
    return g0, g1, g2, g3


@ti.kernel
def semi_euler(
    delta_t: ti.f32,
    pos: ti.template(),
    predict_pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    damping_coeff: ti.f32,
):
    for i in pos:
        vel[i] += delta_t * meta.gravity
        vel[i] *= damping_coeff
        old_pos[i] = pos[i]
        pos[i] += delta_t * vel[i]
        predict_pos[i] = pos[i]


@ti.kernel
def update_vel(delta_t: ti.f32, pos: ti.template(), old_pos: ti.template(), vel: ti.template()):
    for i in pos:
        vel[i] = (pos[i] - old_pos[i]) / delta_t


@ti.kernel
def solve_constraints_kernel(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    inv_mass: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    pos: ti.template(),
    alpha_tilde: ti.template(),
    constraint: ti.template(),
    residual: ti.template(),
    gradC: ti.template(),
    dlambda: ti.template(),
):
    for i in pos:
        pos_mid[i] = pos[i]

    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]

        x0, x1, x2, x3 = pos_mid[p0], pos_mid[p1], pos_mid[p2], pos_mid[p3]

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint[t] = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
        gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0, g1, g2, g3
        denominator = (
            inv_mass[p0] * g0.norm_sqr()
            + inv_mass[p1] * g1.norm_sqr()
            + inv_mass[p2] * g2.norm_sqr()
            + inv_mass[p3] * g3.norm_sqr()
        )
        residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t])
        dlambda[t] = residual[t] / (denominator + alpha_tilde[t])
        lagrangian[t] += dlambda[t]


@ti.kernel
def compute_C_and_gradC_kernel(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    constraint: ti.template(),
    gradC: ti.template(),
):
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]
        x0, x1, x2, x3 = pos_mid[p0], pos_mid[p1], pos_mid[p2], pos_mid[p3]
        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint[t] = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = compute_gradient(U, S, V, B[t])
        # g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
        # g0_ = g0/g0.norm()
        # g1_ = g1/g1.norm()
        # g2_ = g2/g2.norm()
        # g3_ = g3/g3.norm()
        # gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0_, g1_, g2_, g3_



@ti.kernel
def compute_dual_residual(
    constraint: ti.template(),
    alpha_tilde: ti.template(),
    lagrangian: ti.template(),
    dual_residual:ti.template()
):
    for t in range(dual_residual.shape[0]):
        dual_residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t])


@ti.kernel
def project_constraints(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    inv_mass: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    pos: ti.template(),
    alpha_tilde: ti.template(),
    constraint: ti.template(),
    residual: ti.template(),
    gradC: ti.template(),
    dlambda: ti.template(),
    dpos: ti.template(),
):
    for i in pos:
        pos_mid[i] = pos[i]

    # ti.loop_config(serialize=meta.serialize)
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]

        x0, x1, x2, x3 = pos_mid[p0], pos_mid[p1], pos_mid[p2], pos_mid[p3]

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint[t] = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
        gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0, g1, g2, g3
        denorminator = (
            inv_mass[p0] * g0.norm_sqr()
            + inv_mass[p1] * g1.norm_sqr()
            + inv_mass[p2] * g2.norm_sqr()
            + inv_mass[p3] * g3.norm_sqr()
        )
        residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t])
        dlambda[t] = residual[t] / (denorminator + alpha_tilde[t])

        lagrangian[t] += dlambda[t]

    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]
        pos[p0] += meta.omega * inv_mass[p0] * dlambda[t] * gradC[t, 0]
        pos[p1] += meta.omega * inv_mass[p1] * dlambda[t] * gradC[t, 1]
        pos[p2] += meta.omega * inv_mass[p2] * dlambda[t] * gradC[t, 2]
        pos[p3] += meta.omega * inv_mass[p3] * dlambda[t] * gradC[t, 3]
        dpos[p0] += meta.omega * inv_mass[p0] * dlambda[t] * gradC[t, 0]
        dpos[p1] += meta.omega * inv_mass[p1] * dlambda[t] * gradC[t, 1]
        dpos[p2] += meta.omega * inv_mass[p2] * dlambda[t] * gradC[t, 2]
        dpos[p3] += meta.omega * inv_mass[p3] * dlambda[t] * gradC[t, 3]


@ti.kernel
def collsion_response(pos: ti.template()):
    for i in pos:
        ...
        # if pos[i][1] < -1.3:
        #     pos[i][1] = -1.3


def reset_dpos(dpos):
    dpos.fill(0.0)

@ti.kernel
def transfer_back_to_pos_mfree_kernel(gradC:ti.template(),
                                      tet_indices:ti.template(),
                                        inv_mass:ti.template(),
                                        dlambda:ti.template(),
                                        lagrangian:ti.template(),
                                        dpos:ti.template()

):
    for i in range(tet_indices.shape[0]):
        idx0, idx1, idx2, idx3 = tet_indices[i]
        lagrangian[i] += dlambda[i]
        dpos[idx0] += inv_mass[idx0] * dlambda[i] * gradC[i, 0]
        dpos[idx1] += inv_mass[idx1] * dlambda[i] * gradC[i, 1]
        dpos[idx2] += inv_mass[idx2] * dlambda[i] * gradC[i, 2]
        dpos[idx3] += inv_mass[idx3] * dlambda[i] * gradC[i, 3]

@ti.kernel
def update_pos(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(inv_mass.shape[0]):
        if inv_mass[i] != 0.0:
            pos[i] += meta.omega * dpos[i]

def transfer_back_to_pos_mfree(x, ist):
    ist.dlambda.from_numpy(x)
    reset_dpos(ist.dpos)
    transfer_back_to_pos_mfree_kernel(ist.gradC, ist.tet_indices, ist.inv_mass, ist.dlambda, ist.lagrangian, ist.dpos)
    update_pos(ist.inv_mass, ist.dpos, ist.pos)
    collsion_response(ist.pos)

@ti.kernel
def compute_potential_energy(potential_energy:ti.template(),
                             alpha:ti.template(),
                             constraints:ti.template()):
    potential_energy[None] = 0.0
    for i in range(constraints.shape[0]):
        inv_alpha = 1.0/alpha[i]
        potential_energy[None] += 0.5 * inv_alpha * constraints[i]**2

@ti.kernel
def compute_inertial_energy(inertial_energy:ti.template(),
                            inv_mass:ti.template(),
                            pos:ti.template(),
                            predict_pos:ti.template(),
                            delta_t:ti.f32):
    inertial_energy[None] = 0.0
    inv_h2 = 1.0 / delta_t**2
    for i in range(pos.shape[0]):
        if inv_mass[i] == 0.0:
            continue
        inertial_energy[None] += 0.5 / inv_mass[i] * (pos[i] - predict_pos[i]).norm_sqr() * inv_h2


@ti.kernel
def calc_dual_residual(alpha_tilde:ti.template(),
                       lagrangian:ti.template(),
                       constraint:ti.template(),
                       dual_residual:ti.template()):
    for i in range(dual_residual.shape[0]):
        dual_residual[i] = -(constraint[i] + alpha_tilde[i] * lagrangian[i])

def calc_primary_residual(G,M_inv,predict_pos,pos,lagrangian):
    MASS = scipy.sparse.diags(1.0/(M_inv.diagonal()+1e-12), format="csr")
    primary_residual = MASS @ (predict_pos.to_numpy().flatten() - pos.to_numpy().flatten()) - G.transpose() @ lagrangian.to_numpy()
    where_zeros = np.where(M_inv.diagonal()==0)
    primary_residual = np.delete(primary_residual, where_zeros)
    return primary_residual


# To deal with the json dump error for np.float32
# https://ellisvalentiner.com/post/serializing-numpyfloat32-json/
@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

# TODO: DEPRECATE
def fill_A_by_spmm(ist,  M_inv, ALPHA):
    ii, jj, vv = np.zeros(ist.NT*200, dtype=np.int32), np.zeros(ist.NT*200, dtype=np.int32), np.zeros(ist.NT*200, dtype=np.float32)
    fill_gradC_triplets_kernel(ii,jj,vv, ist.gradC, ist.tet_indices)
    G = scipy.sparse.coo_array((vv, (ii, jj)))

    # assemble A
    A = G @ M_inv @ G.transpose() + ALPHA
    A = scipy.sparse.csr_matrix(A, dtype=np.float32)
    # A = scipy.sparse.diags(A.diagonal(), format="csr")
    return A


def csr_is_equal(A, B):
    if A.shape != B.shape:
        print("shape not equal")
        assert False
    diff = A - B
    if diff.nnz == 0:
        print("csr is equal! nnz=0")
        return True
    maxdiff = np.abs(diff.data).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        assert False
    print("csr is equal!")
    return True


def calc_dual(ist):
    calc_dual_residual(ist.dual_residual, ist.lagrangian, ist.constraint, ist.dual_residual)
    return ist.dual_residual.to_numpy()


def AMG_A():
    tic2 = perf_counter()
    # A = fill_A_csr_ti(ist)
    extlib.fastFillSoft_run(ist.pos.to_numpy(), ist.gradC.to_numpy())
    logging.info(f"    fill_A time: {(perf_counter()-tic2)*1000:.0f}ms")
    # return A


def AMG_b(ist):
    b = -ist.constraint.to_numpy() - ist.alpha_tilde_np * ist.lagrangian.to_numpy()
    return b


def should_setup():
    return ((meta.frame%args.setup_interval==0 or (args.restart==True and meta.frame==args.restart_frame)) and (meta.ite==0))

def update_P(Ps):
    for lv in range(len(Ps)):
        P_ = Ps[lv].tocsr()
        extlib.fastmg_set_P(lv, P_.data.astype(np.float32), P_.indices, P_.indptr, P_.shape[0], P_.shape[1], P_.nnz)


def cuda_set_A0(A0):
    extlib.fastmg_set_A0(A0.data.astype(np.float32), A0.indices, A0.indptr, A0.shape[0], A0.shape[1], A0.nnz)


def report_multilevel_details(Ps, num_levels):
    logging.info(f"    num_levels:{num_levels}")
    num_points_level = []
    for i in range(len(Ps)):
        num_points_level.append(Ps[i].shape[0])
    num_points_level.append(Ps[-1].shape[1])
    for i in range(num_levels):
        logging.info(f"    num points of level {i}: {num_points_level[i]}")


def smoother_name2type(name):
    if name == "chebyshev":
        return 1
    elif name == "jacobi":
        return 2
    elif name == "gauss_seidel":
        return 3
    else:
        raise ValueError(f"smoother name {name} not supported")

def AMG_setup_phase():
    global Ps
    tic = time.perf_counter()
    A = fill_A_csr_ti(ist) #taichi version
    A = A.copy() #FIXME: no copy will cause bug, why?
    Ps = build_Ps(A)
    logging.info(f"    build_Ps time:{time.perf_counter()-tic}")


    tic = time.perf_counter()
    update_P(Ps)
    logging.info(f"    update_P time: {time.perf_counter()-tic:.2f}s")

    tic = time.perf_counter()
    cuda_set_A0(A)
    
    AMG_RAP()

    s = smoother_name2type(args.smoother_type)
    extlib.fastmg_setup_smoothers.argtypes = [c_int]
    print(s)
    extlib.fastmg_setup_smoothers(s) # 1 means chebyshev, 2 means w-jacobi, 3 gauss_seidel
    extlib.fastmg_set_smoother_niter(args.smoother_niter)
    logging.info(f"    setup smoothers time:{perf_counter()-tic}")

    if use_graph_coloring:
        graph_coloring_v2()    
    return A


def fetch_A_from_cuda(lv=0):
    extlib.fastmg_get_nnz.argtypes = [ctypes.c_int]
    extlib.fastmg_get_nnz.restype = ctypes.c_int
    extlib.fastmg_get_matsize.argtypes = [ctypes.c_int]
    extlib.fastmg_get_matsize.restype = ctypes.c_int
    extlib.fastmg_fetch_A.argtypes = [c_int, arr_float, arr_int, arr_int]

    nnz = extlib.fastmg_get_nnz(lv)
    matsize = extlib.fastmg_get_matsize(lv)

    A_data = np.zeros(nnz, dtype=np.float32)
    A_indices = np.zeros(nnz, dtype=np.int32)
    A_indptr = np.zeros(matsize+1, dtype=np.int32)

    extlib.fastmg_fetch_A(lv, A_data, A_indices, A_indptr)
    A = scipy.sparse.csr_matrix((A_data, A_indices, A_indptr), shape=(matsize, matsize))
    return A


def fastmg_fetch():
    extlib.fastmg_fetch_A_data.argtypes = [arr_float]
    extlib.fastmg_fetch_A_data(ist.data)
    A = scipy.sparse.csr_matrix((ist.data, ist.indices, ist.indptr), shape=(ist.NT, ist.NT))
    return A


def AMG_RAP():
    tic3 = time.perf_counter()
    # A = fill_A_csr_ti(ist)
    # cuda_set_A0(A)
    for lv in range(num_levels-1):
        extlib.fastmg_RAP(lv) 
    logging.info(f"    RAP time: {(time.perf_counter()-tic3)*1000:.0f}ms")


def AMG_dlam2dpos(x):
    tic = time.perf_counter()
    transfer_back_to_pos_mfree(x, ist)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")


def AMG_solve(b, x0=None, tol=1e-5, maxiter=100):
    if x0 is None:
        x0 = np.zeros(b.shape[0], dtype=np.float32)

    tic4 = time.perf_counter()
    # set data
    x0 = x0.astype(np.float32)
    b = b.astype(np.float32)
    extlib.fastmg_set_data(x0, x0.shape[0], b, b.shape[0], tol, maxiter)

    # solve
    if args.only_smoother:
        extlib.fastmg_solve_only_smoother()
    else:
        extlib.fastmg_solve()

    # get result
    x = np.empty_like(x0, dtype=np.float32)
    residuals = np.zeros(shape=(maxiter,), dtype=np.float32)
    niter = extlib.fastmg_get_data(x, residuals)
    niter += 1
    residuals = residuals[:niter]
    logging.info(f"    inner iter: {niter}")
    logging.info(f"    solve time: {(time.perf_counter()-tic4)*1000:.0f}ms")
    return (x),  residuals  


def do_export_r(r):
    global t_export
    tic = time.perf_counter()
    serialized_r = [r[i]._asdict() for i in range(len(r))]
    r_json = json.dumps(serialized_r)
    with open(out_dir+'/r/'+ f'{meta.frame}.json', 'w') as file:
        file.write(r_json)
    t_export += time.perf_counter()-tic


def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))


def AMG_calc_r(r,fulldual0, tic_iter, r_Axb):
    global t_export
    tic = time.perf_counter()

    t_iter = perf_counter()-tic_iter
    tic_calcr = perf_counter()
    calc_dual_residual(ist.alpha_tilde, ist.lagrangian, ist.constraint, ist.dual_residual)
    dual_r = np.linalg.norm(ist.dual_residual.to_numpy()).astype(float)
    r_Axb = r_Axb.tolist()
    dual0 = np.linalg.norm(fulldual0)

    logging.info(f"    convergence factor: {calc_conv(r_Axb):.2g}")
    logging.info(f"    Calc r time: {(perf_counter()-tic_calcr)*1000:.0f}ms")

    if args.export_log:
        logging.info(f"    iter total time: {t_iter*1000:.0f}ms")
        logging.info(f"{meta.frame}-{meta.ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual0:{dual0:.2e} dual:{dual_r:.2e} iter:{len(r_Axb)}")
    r.append(ResidualData(dual_r, len(r_Axb), t_iter))

    t_export += perf_counter()-tic
    return dual0


def AMG_python(b):
    global Ps, num_levels

    A = fill_A_csr_ti(ist)
    A = A.copy()#FIXME: no copy will cause bug, why?

    if should_setup():
        tic = time.perf_counter()
        Ps = build_Ps(A)
        num_levels = len(Ps)+1
        logging.info(f"    build_Ps time:{time.perf_counter()-tic}")
    
    tic = time.perf_counter()
    levels = build_levels(A, Ps)
    logging.info(f"    build_levels time:{time.perf_counter()-tic}")

    if should_setup():
        tic = time.perf_counter()
        setup_smoothers(A)
        logging.info(f"    setup smoothers time:{perf_counter()-tic}")
    x0 = np.zeros_like(b)
    tic = time.perf_counter()
    x, r_Axb = old_amg_cg_solve(levels, b, x0=x0, maxiter=args.maxiter_Axb, tol=1e-6)
    toc = time.perf_counter()
    logging.info(f"    mgsolve time {toc-tic}")
    return  x, r_Axb




class AMGXSolver:
    def __init__(self, config_file):
        self.config_file = config_file
        amgx_lib_dir = "D:/Dev/AMGX/build/Release"
        cuda_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
        os.add_dll_directory(amgx_lib_dir)
        os.add_dll_directory(cuda_dir)
        import pyamgx
        self.pyamgx = pyamgx

    def update(self, data, rhs):
        self.A.replace_coefficients(data.astype(np.float64))
        self.b.upload(rhs.astype(np.float64))
        
        self.solver.setup(self.A)
        self.solver.solve(self.b, self.x)
        self.niter = self.solver.iterations_number
        self.status = self.solver.status

        # assert self.status == 'success'
        logging.info("pyamgx status: ", self.status)
        self.r_Axb = []
        for i in range(self.niter):
            self.r_Axb.append(self.solver.get_residual(i))
        self.x.download(self.sol)
        return self.sol, self.r_Axb, self.niter
    
    def init(self):
        self.pyamgx.initialize()
        self.cfg = self.pyamgx.Config().create_from_file(self.config_file)
        self.rsc = self.pyamgx.Resources().create_simple(self.cfg)
        # Create matrices and vectors:
        self.A = self.pyamgx.Matrix().create(self.rsc)
        self.b = self.pyamgx.Vector().create(self.rsc)
        self.x = self.pyamgx.Vector().create(self.rsc)
        # Create solver:
        self.solver = self.pyamgx.Solver().create(self.rsc, self.cfg)

    def solve(self, M, rhs):
        # self.pyamgx.initialize()
        # self.cfg = self.pyamgx.Config().create_from_file(self.config_file)
        # self.rsc = self.pyamgx.Resources().create_simple(self.cfg)
        # # Create matrices and vectors:
        # self.A = self.pyamgx.Matrix().create(self.rsc)
        # self.b = self.pyamgx.Vector().create(self.rsc)
        # self.x = self.pyamgx.Vector().create(self.rsc)
        # # Create solver:
        # self.solver = self.pyamgx.Solver().create(self.rsc, self.cfg)

        # Upload system:
        self.M = M.astype(np.float64)
        self.rhs = rhs.astype(np.float64)
        self.sol = np.zeros(rhs.shape[0], dtype=np.float64)
        self.A.upload_CSR(self.M)
        self.b.upload(self.rhs)
        self.x.upload(self.sol)

        # Setup and solve system:
        # if should_setup():
        self.solver.setup(self.A)
        self.solver.solve(self.b, self.x)
        self.niter = self.solver.iterations_number


        self.r_Axb = []
        for i in range(self.niter):
            self.r_Axb.append(self.solver.get_residual(i))
        # self.r_final = self.solver.get_residual()
        # self.r0 = self.solver.get_residual(0)

        self.x.download(self.sol)
        
        status = self.solver.status
        # assert status == 'success', f"status:{status}, iterations: {self.niter}. The residual is {self.r_Axb}"
        if status != 'success':
            logging.info(f"status:{status}, iterations: {self.niter}. The residual is {self.r_Axb[-1]}")

        return self.sol, self.r_Axb, self.niter

    def finalize(self):
        # Clean up:
        self.A.destroy()
        self.x.destroy()
        self.b.destroy()
        self.solver.destroy()
        self.rsc.destroy()
        self.pyamgx.finalize()


has_init_amgx = False
def AMG_amgx(b):
    tic = time.perf_counter()
    global has_init_amgx, amgxsolver
    if not has_init_amgx:
        # config_file = Path(prj_path + "/data/config/AGGREGATION_JACOBI.json")
        # config_file = Path(prj_path + "/data/config/agg_cheb4.json")
        # config_file = str(config_file)
        amgxsolver = AMGXSolver(args.amgx_config)
        amgxsolver.init()
        has_init_amgx = True
    
    A = fill_A_csr_ti(ist)
    A = A.copy()#FIXME: no copy will cause bug, why?
    # x, r_Axb, niter = amgxsolver.update(A.data, b)
    x, r_Axb, niter = amgxsolver.solve(A, b)
    # amgxsolver.finalize()
    logging.info(f"    AMGX time: {(time.perf_counter()-tic)*1000:.0f}ms")
    return x, np.array(r_Axb)



def AMG_cuda(b):
    AMG_A()
    if should_setup():
        A = AMG_setup_phase()
        if args.export_matrix:
            export_A_b(A, b, postfix=f"F{meta.frame}",binary=args.export_matrix_binary)
    extlib.fastmg_set_A0_from_fastFillSoft()
    AMG_RAP()
    x, r_Axb = AMG_solve(b, maxiter=args.maxiter_Axb, tol=args.tol_Axb)
    return x, r_Axb


def substep_all_solver(ist):
    global t_export, n_outer_all
    tic1 = time.perf_counter()
    semi_euler(meta.delta_t, ist.pos, ist.predict_pos, ist.old_pos, ist.vel, meta.damping_coeff)
    reset_lagrangian(ist.lagrangian)
    r = [] # residual list of one frame
    logging.info(f"pre-loop time: {(perf_counter()-tic1)*1000:.0f}ms")
    for meta.ite in range(args.maxiter):
        tic_iter = perf_counter()
        ist.pos_mid.from_numpy(ist.pos.to_numpy())
        compute_C_and_gradC_kernel(ist.pos_mid, ist.tet_indices, ist.B, ist.constraint, ist.gradC)
        if meta.ite==0:
            fulldual0 = calc_dual(ist)
        b = AMG_b(ist)
        if not args.use_cuda:
            x, r_Axb = AMG_python(b)
        else:
            if args.solver_type == "AMG":
                x, r_Axb = AMG_cuda(b)
            elif args.solver_type == "AMGX":
                x, r_Axb = AMG_amgx(b)
        AMG_dlam2dpos(x)
        dual0 = AMG_calc_r(r, fulldual0, tic_iter, r_Axb)
        logging.info(f"iter time(with export): {(perf_counter()-tic_iter)*1000:.0f}ms")
        if r[-1].dual<args.tol:
            break
        if is_stall(r):
            logging.info("Stall detected, break")
            break
        if r[-1].dual / r[0].dual <args.rtol:
            break
        if is_diverge(r, r_Axb):
            logging.error("Diverge detected, break")
            break
    
    tic = time.perf_counter()
    logging.info(f"n_outer: {len(r)}")
    n_outer_all.append(len(r))
    if args.export_residual:
        do_export_r(r)
    collsion_response(ist.pos)
    update_vel(meta.delta_t, ist.pos, ist.old_pos, ist.vel)
    logging.info(f"post-loop time: {(time.perf_counter()-tic)*1000:.0f}ms")
    t_avg_iter.append((time.perf_counter()-tic1)/n_outer_all[-1])
    logging.info(f"avg iter frame {meta.frame}: {t_avg_iter[-1]*1000:.0f}ms")


all_stalled = []
# if in last 5 iters, residuals not change 0.1%, then it is stalled
def is_stall(r):
    if (meta.ite < 5):
        return False
    # a=np.array([r[-1].dual, r[-2].dual,r[-3].dual,r[-4].dual,r[-5].dual])
    inc1 = r[-1].dual/r[-2].dual
    inc2 = r[-2].dual/r[-3].dual
    inc3 = r[-3].dual/r[-4].dual
    inc4 = r[-4].dual/r[-5].dual
    
    # if all incs is in [0.999,1.001]
    if np.all((inc1>0.999) & (inc1<1.001) & (inc2>0.999) & (inc2<1.001) & (inc3>0.999) & (inc3<1.001) & (inc4>0.999) & (inc4<1.001)):
        logging.warning(f"Stall at {meta.frame}-{meta.ite}")
        all_stalled.append((meta.frame, meta.ite))
        return True
    return False


def is_diverge(r,r_Axb):
    if (meta.ite < 5):
        return False

    if r[-1].dual/r[-5].dual>5:
        return True
    
    if r_Axb[-1]>r_Axb[0]:
        return True

    return False


def substep_xpbd(ist):
    global n_outer_all
    semi_euler(meta.delta_t, ist.pos, ist.predict_pos, ist.old_pos, ist.vel, meta.damping_coeff)
    reset_lagrangian(ist.lagrangian)
    r=[]
    for meta.ite in range(args.maxiter):
        tic = time.perf_counter()
        project_constraints(
            ist.pos_mid,
            ist.tet_indices,
            ist.inv_mass,
            ist.lagrangian,
            ist.B,
            ist.pos,
            ist.alpha_tilde,
            ist.constraint,
            ist.residual,
            ist.gradC,
            ist.dlambda,
            ist.dpos,
        )
        collsion_response(ist.pos)
        calc_dual_residual(ist.alpha_tilde, ist.lagrangian, ist.constraint, ist.dual_residual)
        dualr = np.linalg.norm(ist.residual.to_numpy())
        if meta.ite == 0:
            dualr0 = dualr.copy()
        toc = time.perf_counter()
        logging.info(f"{meta.frame}-{meta.ite} dual0:{dualr0:.2e} dual:{dualr:.2e} t:{toc-tic:.2e}s")
        r.append(ResidualData(dualr, 0, toc-tic))
        if dualr < args.tol:
            logging.info("Converge: tol")
            break
        if dualr / dualr0 < args.rtol:
            logging.info("Converge: rtol")
            break
        if is_stall(r):
            logging.warning("Stall detected, break")
            break
    n_outer_all.append(meta.ite+1)
    update_vel(meta.delta_t, ist.pos, ist.old_pos, ist.vel)

# ---------------------------------------------------------------------------- #
#                                    amgpcg                                    #
# ---------------------------------------------------------------------------- #

chebyshev_coeff = None
def chebyshev(A, x, b, coefficients=chebyshev_coeff, iterations=1):
    x = np.ravel(x)
    b = np.ravel(b)
    for _i in range(iterations):
        residual = b - A*x
        h = coefficients[0]*residual
        for c in coefficients[1:]:
            h = c*residual + A*h
        x += h


def setup_chebyshev_python(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3,
                    iterations=1):
    global chebyshev_coeff 
    """Set up Chebyshev."""
    rho = approximate_spectral_radius(A)
    a = rho * lower_bound
    b = rho * upper_bound
    # drop the constant coefficient
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]
    chebyshev_coeff = coefficients
    return coefficients


def setup_jacobi_python(A):
    from pyamg.relaxation.smoothing import rho_D_inv_A
    global jacobi_omega
    tic = perf_counter()
    rho = rho_D_inv_A(A)
    print("rho:", rho)
    jacobi_omega = 1.0/(rho)
    print("omega:", jacobi_omega)
    toc = perf_counter()
    print("Calculating jacobi omega Time:", toc-tic)
    return jacobi_omega


def calc_near_nullspace_GS(A):
    n=6
    print("Calculating near nullspace")
    tic = perf_counter()
    B = np.zeros((A.shape[0],n), dtype=np.float64)
    from pyamg.relaxation.relaxation import gauss_seidel
    for i in range(n):
        x = np.ones(A.shape[0]) + 1e-2*np.random.rand(A.shape[0])
        b = np.zeros(A.shape[0]) 
        gauss_seidel(A,x.astype(np.float32),b.astype(np.float32),iterations=20, sweep='forward')
        B[:,i] = x
        print(f"norm B {i}: {np.linalg.norm(B[:,i])}")
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B


def do_filter_P(P, theta=0.25):
    # filter out the small values in each column of P
    # small value: |val| < 0.25 |max_val|
    logging.info(f"Filtering P, shape: {P.shape}")
    P = P.tocsc()
    indices, indptr, data = P.indices, P.indptr, P.data
    for j in range(P.shape[1]):
        col_start = indptr[j]
        col_end = indptr[j + 1]
        col_data = data[col_start:col_end]
        max_val = np.abs(col_data).max()
        ...
        for i in range(col_start, col_end):
            if np.abs(data[i]) < theta * max_val:
                data[i] = 0
    P.eliminate_zeros()
    return P.tocsr()


def do_set_01_P(P):
    # for all non-zero values in P, set them to 1
    logging.info(f"set 01 P, shape: {P.shape}")
    P.data[:] = 1
    P = P.tocsr()
    logging.info(f"set 01 P done")
    return P


def do_set_avg_P(P):
    # for all non-zero values in P, set them to 1
    logging.info(f"set avg P, shape: {P.shape}")
    P.data[:] = 1
    # for each column, set the each value to avg, and sum to 1.0
    P = P.tocsc()
    for j in range(P.shape[1]):
        col_start = P.indptr[j]
        col_end = P.indptr[j + 1]
        col_data = P.data[col_start:col_end]
        col_sum = np.sum(col_data)
        if col_sum != 0:
            P.data[col_start:col_end] /= col_sum
    P = P.tocsr()
    logging.info(f"set avg P done")
    return P


def calc_RAP_scale(P):
    logging.info(f"get RAP scale from nnz of each column of P, shape: {P.shape}")
    P = P.tocsc()
    nnz_col = np.zeros(P.shape[1], dtype=np.int32)
    for j in range(P.shape[1]):
        col_start = P.indptr[j]
        col_end = P.indptr[j + 1]
        nnz_col[j] = col_end - col_start #size of aggregate
    scale = 1.0/nnz_col.mean()
    logging.info(f"RAP scale={scale}")
    return scale


def build_Ps(A):
    """Build a list of prolongation matrices Ps from A """
    method = args.build_P_method
    print("build P by method:", method)
    tic = perf_counter()
    if method == 'UA':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None, symmetry='symmetric')
    elif method == 'SA' :
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400,symmetry='symmetric')
    elif method == 'CAMG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400)
    elif method == 'adaptive_SA':
        ml = pyamg.aggregation.adaptive_sa_solver(A.astype(np.float64), max_coarse=400, smooth=None, num_candidates=6)[0]
    elif method == 'nullspace':
        B = calc_near_nullspace_GS(A)
        print("B shape:", B.shape)
        print(f"B: {B}")
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None,symmetry='symmetric', B=B)
    elif method == 'algebraic3.0':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('algebraic_distance', {'epsilon': 3.0}))
    elif method == 'affinity4.0':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('affinity', {'epsilon': 4.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    elif method == 'strength0.1':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.1 }))    
    elif method == 'strength0.2':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.2 }))    
    elif method == 'strength0.25':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.25 }))
    elif method == 'strength0.3':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.3 }))
    elif method == 'strength0.4':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.4 }))
    elif method == 'strength0.5':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.5 }))
    elif method == 'evolution':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('evolution', {'k': 2, 'proj_type': 'l2', 'epsilon': 4.0}))
    elif method == 'improve_candidate':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth = None, improve_candidates=(('block_gauss_seidel',{'sweep': 'symmetric','iterations': 4}),None), symmetry='symmetric', strength=('symmetric',{'theta' : 0.1 }))
    elif method == 'strength_energy':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('energy_based',{'theta' : 0.25 })) 
    elif method == 'strength_classical':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('classical')) 
    elif method == 'strength_distance':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('distance')) 
    elif method == 'aggregate_standard':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='standard')
    elif method == 'aggregate_naive':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='naive')
    elif method == 'aggregate_lloyd':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='lloyd')
    elif method == 'aggregate_pairwise':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='pairwise')
    elif method == 'diagonal_dominance':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.1 }),diagonal_dominance=True)
    else:
        raise ValueError(f"Method {method} not recognized")

    global num_levels
    num_levels = len(ml.levels)
    extlib.fastmg_setup_nl.argtypes = [ctypes.c_size_t]
    extlib.fastmg_setup_nl(num_levels)
    
    logging.info(ml)

    Ps = []
    for i in range(len(ml.levels)-1):
        P = ml.levels[i].P
        if args.filter_P=="fileter":
            P = do_filter_P(P,0.25)
        elif args.filter_P=="01":
            P = do_set_01_P(P)
        elif args.filter_P=="avg":
            P = do_set_avg_P(P)
        Ps.append(P)

        if args.scale_RAP:
            # scale RAP by avg size of aggregates
            # get scale from nnz of each column of P
            s = calc_RAP_scale(P)
            extlib.fastmg_scale_RAP(s, i)



    toc = perf_counter()
    logging.info(f"Build P Time:{toc-tic:.2f}s")

    logger2.info(f"logger2 {method} {toc-tic}")
    # file = out_dir+'/build_P_time.txt'
    # with open(file, 'a') as f:
    #     f.write(f"{method} {toc-tic}\n")
    return Ps


class MultiLevel:
    A = None
    P = None
    R = None


def build_levels(A, Ps=[]):
    '''Give A and a list of prolongation matrices Ps, return a list of levels'''
    lvl = len(Ps) + 1 # number of levels

    levels = [MultiLevel() for i in range(lvl)]

    levels[0].A = A

    for i in range(lvl-1):
        levels[i].P = Ps[i]
        levels[i].R = Ps[i].T
        levels[i+1].A = Ps[i].T @ levels[i].A @ Ps[i]

    return levels


def setup_smoothers(A):
    global chebyshev_coeff
    if args.smoother_type == 'chebyshev':
        setup_chebyshev_python(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3)
    elif args.smoother_type == 'jacobi':
        setup_jacobi_python(A)


def old_amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    assert x0 is not None
    x = x0.copy()
    A = levels[0].A
    residuals = np.zeros(maxiter+1)
    def psolve(b):
        x = x0.copy()
        old_V_cycle(levels, 0, x, b)
        return x
    bnrm2 = np.linalg.norm(b)
    atol = tol * bnrm2
    r = b - A@(x)
    rho_prev, p = None, None
    normr = np.linalg.norm(r)
    residuals[0] = normr
    iteration = 0
    for iteration in range(maxiter):
        if normr < atol:  # Are we done?
            break
        z = psolve(r)
        rho_cur = np.dot(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]
        q = A@(p)
        alpha = rho_cur / np.dot(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur
        normr = np.linalg.norm(r)
        residuals[iteration+1] = normr
    residuals = residuals[:iteration+1]
    return (x),  residuals  


def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag

def presmoother(A,x,b):
    A = A.astype(np.float32)
    from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial, schwarz
    if args.smoother_type == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif args.smoother_type == 'jacobi':
        jacobi(A,x,b,iterations=10, omega=jacobi_omega)
    elif args.smoother_type == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif args.smoother_type == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif args.smoother_type == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif args.smoother_type == 'chebyshev':
        chebyshev(A,x,b)
    elif args.smoother_type == 'schwarz':
        schwarz(A,x,b)


def postsmoother(A,x,b):
    presmoother(A,x,b)


def coarse_solver(A, b):
    res = np.linalg.solve(A.toarray(), b)
    return res


def old_V_cycle(levels,lvl,x,b):
    A = levels[lvl].A.astype(np.float64)
    presmoother(A,x,b)
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        old_V_cycle(levels, lvl+1, coarse_x, coarse_b)
    x += levels[lvl].P @ coarse_x
    postsmoother(A, x, b)


# ---------------------------------------------------------------------------- #
#                                  amgpcg end                                  #
# ---------------------------------------------------------------------------- #
def make_and_clean_dirs(out_dir):
    import shutil
    from pathlib import Path

    shutil.rmtree(out_dir, ignore_errors=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/r/").mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/A/").mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/state/").mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/mesh/").mkdir(parents=True, exist_ok=True)


def export_A_b(A,b,postfix="", binary=True):
    dir = out_dir + "/A/"
    logging.info(f"Exporting A and b to {dir}...")
    if binary:
        scipy.sparse.save_npz(dir + f"A_{postfix}.npz", A)
        np.save(dir + f"b_{postfix}.npy", b)
        # A = scipy.sparse.load_npz("A.npz") # load
    else:
        scipy.io.mmwrite(dir + f"A_{postfix}.mtx", A, symmetry='symmetric')
        np.savetxt(dir + f"b_{postfix}.txt", b)


def use_another_outdir(out_dir):
    import re
    path = Path(out_dir)
    if path.exists():
        # 使用正则表达式匹配文件夹名称中的数字后缀
        base_name = path.name
        match = re.search(r'_(\d+)$', base_name)
        if match:
            base_name = base_name[:match.start()]
            i = int(match.group(1)) + 1
        else:
            base_name = base_name
            i = 1

        while True:
            new_name = f"{base_name}_{i}"
            path = path.parent / new_name
            if not path.exists():
                break
            i += 1

    out_dir = str(path)
    print(f"\nFind another outdir: {out_dir}\n")
    return out_dir


def ending(timer_loop, start_date, initial_frame):
    global n_outer_all, t_export_total, all_stalled
    t_all = time.perf_counter() - timer_loop
    end_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.end_frame = meta.frame

    len_n_outer_all = len(n_outer_all) if len(n_outer_all) > 0 else 1
    sum_n_outer = sum(n_outer_all)
    avg_n_outer = sum_n_outer / len_n_outer_all
    max_n_outer = max(n_outer_all)
    max_n_outer_index = n_outer_all.index(max_n_outer)

    n_outer_all_np = np.array(n_outer_all, np.int32)    
    np.savetxt(out_dir+"/n_outer.txt", n_outer_all_np, fmt="%d")

    sim_time_with_export = time.perf_counter() - timer_loop
    sim_time = sim_time_with_export - t_export_total
    avg_sim_time = sim_time / (args.end_frame - initial_frame)


    s = f"\n-------\n"+\
    f"Time: {(sim_time):.2f}s = {(sim_time)/60:.2f}min.\n" + \
    f"Time with exporting: {(sim_time_with_export):.2f}s = {sim_time_with_export/60:.2f}min.\n" + \
    f"Frame {initial_frame}-{args.end_frame}({args.end_frame-initial_frame} frames)."+\
    f"\nAvg: {avg_sim_time}s/frame."+\
    f"\nStart\t{start_date},\nEnd\t{end_date}."+\
    f"\nTime of exporting: {t_export_total:.3f}s" + \
    f"\nSum n_outer: {sum_n_outer} \nAvg n_outer: {avg_n_outer:.1f}"+\
    f"\nMax n_outer: {max_n_outer} \nMax n_outer frame: {max_n_outer_index + initial_frame}." + \
    f"\nstalled at {all_stalled}"+\
    f"\nmodel_path: {args.model_path}" + \
    f"\ndt={meta.delta_t}" + \
    f"\nSolver: {args.solver_type}" + \
    f"\nout_dir: {out_dir}" 
    # logging.info(s)


    file_name = f"result/meta/{out_dir_name}.log"
    file_name2 = f"{out_dir}/meta.log"
    logger_meta = logging.getLogger('logger_meta')
    logger_meta.addHandler(logging.FileHandler(file_name))
    logger_meta.addHandler(logging.FileHandler(file_name2))
    logger_meta.info(s)

    if args.solver_type == "AMGX":
        amgxsolver.finalize()

# ---------------------------------------------------------------------------- #
#                               directly  fill A                               #
# ---------------------------------------------------------------------------- #
def init_adj_ele(eles):
    vertex_to_eles = {}
    for ele_index, (v1, v2, v3, v4) in enumerate(eles):
        if v1 not in vertex_to_eles:
            vertex_to_eles[v1] = set()
        if v2 not in vertex_to_eles:
            vertex_to_eles[v2] = set()
        if v3 not in vertex_to_eles:
            vertex_to_eles[v3] = set()
        if v4 not in vertex_to_eles:
            vertex_to_eles[v4] = set()
        
        vertex_to_eles[v1].add(ele_index)
        vertex_to_eles[v2].add(ele_index)
        vertex_to_eles[v3].add(ele_index)
        vertex_to_eles[v4].add(ele_index)

    all_adjacent_eles = {}

    for ele_index in range(len(eles)):
        v1, v2, v3, v4 = eles[ele_index]
        adjacent_eles = vertex_to_eles[v1] | vertex_to_eles[v2] | vertex_to_eles[v3] | vertex_to_eles[v4]
        adjacent_eles.remove(ele_index)  # 移除本身
        all_adjacent_eles[ele_index] = list(adjacent_eles)
    return all_adjacent_eles, vertex_to_eles


def init_adj_ele_ti(eles):
    eles = eles
    nele = eles.shape[0]
    v2e = ti.field(dtype=ti.i32, shape=(nele, 200))
    nv2e = ti.field(dtype=ti.i32, shape=nele)

    @ti.kernel
    def calc_vertex_to_eles_kernel(eles: ti.template(), v2e: ti.template(), nv2e: ti.template()):
        # v2e: vertex to element
        # nv2e: number of elements sharing the vertex
        for e in range(eles.shape[0]):
            v1, v2, v3, v4 = eles[e]
            for v in ti.static([v1, v2, v3, v4]):
                k = nv2e[v]
                v2e[v, k] = e
                nv2e[v] += 1

    calc_vertex_to_eles_kernel(eles, v2e, nv2e)
    # v2e = v2e.to_numpy()
    # nv2e = nv2e.to_numpy()

# transfer one-to-multiple map dict to ndarray
def dict_to_ndarr(d:dict)->np.ndarray:
    lengths = np.array([len(v) for v in d.values()])
    max_len = max(lengths)
    arr = np.ones((len(d), max_len), dtype=np.int32) * (-1)
    for i, (k, v) in enumerate(d.items()):
        arr[i, :len(v)] = v
    return arr, lengths


def init_A_CSR_pattern(num_adj, adj):
    nrows = len(num_adj)
    nonz = np.sum(num_adj)+nrows
    indptr = np.zeros(nrows+1, dtype=np.int32)
    indices = np.zeros(nonz, dtype=np.int32)
    data = np.zeros(nonz, dtype=np.float32)
    indptr[0] = 0
    for i in range(0,nrows):
        num_adj_i = num_adj[i]
        indptr[i+1]=indptr[i] + num_adj_i + 1
        indices[indptr[i]:indptr[i+1]-1]= adj[i][:num_adj_i]
        indices[indptr[i+1]-1]=i
    assert indptr[-1] == nonz
    return data, indices, indptr


def csr_index_to_coo_index(indptr, indices):
    ii, jj = np.zeros_like(indices), np.zeros_like(indices)
    nrows = len(indptr)-1
    for i in range(nrows):
        ii[indptr[i]:indptr[i+1]]=i
    jj[:]=indices[:]
    return ii, jj


def initFill_tocuda(ist):
    extlib.fastFillSoft_init_from_python_cache_lessmem.argtypes = [c_int]*2  + [arr_float] + [arr_int]*3 + [c_int]

    extlib.fastFillSoft_init_from_python_cache_lessmem(
            ist.NT,
            ist.MAX_ADJ,
            ist.data,
            ist.indices,
            ist.indptr,
            ist.ii,
            ist.nnz)
    extlib.fastFillSoft_set_data(ist.tet_indices.to_numpy(), ist.NT, ist.inv_mass.to_numpy(), ist.NV, ist.pos.to_numpy(), ist.alpha_tilde.to_numpy())


def mem_usage():
    # 内存占用
    # 将字节转换为GB
    def bytes_to_gb(bytes):
        return bytes / (1024 ** 3)

    data_memory_gb = bytes_to_gb(ist.data.nbytes)
    indices_memory_gb = bytes_to_gb(ist.indices.nbytes)
    indptr_memory_gb = bytes_to_gb(ist.indptr.nbytes)
    ii_memory_gb = bytes_to_gb(ist.ii.nbytes)
    total_memory_gb = (data_memory_gb + indices_memory_gb + indptr_memory_gb + ii_memory_gb)

    # 打印每个数组的内存占用和总内存占用（GB）
    print(f"data memory: {data_memory_gb:.2f} GB")
    print(f"indices memory: {indices_memory_gb:.2f} GB")
    print(f"indptr memory: {indptr_memory_gb:.2f} GB")
    print(f"ii memory: {ii_memory_gb:.2f} GB")
    print(f"Total memory: {total_memory_gb:.2f} GB")


def init_direct_fill_A(ist):
    cache_file_name = f'cache_initFill_{os.path.basename(args.model_path)}.npz'
    if args.use_cache and os.path.exists(cache_file_name):
        tic = perf_counter()
        print(f"Found cache {cache_file_name}. Loading cached data...")
        npzfile = np.load(cache_file_name)
        ist.data = npzfile['data']
        ist.indices = npzfile['indices']
        ist.indptr = npzfile['indptr']
        ist.ii = npzfile['ii']
        ist.nnz = int(npzfile['nnz'])
        ist.jj = ist.indices # No need to save jj,  indices is the same as jj
        ist.MAX_ADJ = int(npzfile['MAX_ADJ'])
        print(f"MAX_ADJ: {ist.MAX_ADJ}")
        mem_usage()
        if args.use_cuda:
            initFill_tocuda(ist)
        print(f"Loading cache time: {perf_counter()-tic:.3f}s")
        return

    print(f"No cached data found, initializing...")

    tic1 = perf_counter()
    print("Initializing adjacent elements and abc...")
    adjacent, v2e = init_adj_ele(eles=ist.tet_indices.to_numpy())
    # adjacent = init_adj_ele_ti(eles=ist.tet_indices)
    num_adjacent = np.array([len(v) for v in adjacent.values()])
    AVG_ADJ = np.mean(num_adjacent)
    ist.MAX_ADJ = max(num_adjacent)
    print(f"MAX_ADJ: {ist.MAX_ADJ}")
    print(f"AVG_ADJ: {AVG_ADJ}")
    print(f"init_adjacent time: {perf_counter()-tic1:.3f}s")

    tic = perf_counter()
    ist.data, ist.indices, ist.indptr = init_A_CSR_pattern(num_adjacent, adjacent)
    ist.ii, ist.jj = csr_index_to_coo_index(ist.indptr, ist.indices)
    ist.nnz = len(ist.data)
    # nnz_each_row = num_adjacent[:] + 1
    print(f"init_A_CSR_pattern time: {perf_counter()-tic:.3f}s")
    
    tic = perf_counter()
    adjacent,_ = dict_to_ndarr(adjacent)
    print(f"dict_to_ndarr time: {perf_counter()-tic:.3f}s")

    tic = perf_counter()
    print(f"init_adj_share_v time: {perf_counter()-tic:.3f}s")
    print(f"initFill done")

    mem_usage()

    if args.use_cache:
        print(f"Saving cache to {cache_file_name}...")
        np.savez(cache_file_name, data=ist.data, indices=ist.indices, indptr=ist.indptr, ii=ist.ii, nnz=ist.nnz, MAX_ADJ=ist.MAX_ADJ)
        print(f"{cache_file_name} saved")
    if args.use_cuda:
        initFill_tocuda(ist)


def fill_A_csr_ti(ist):
    fill_A_csr_lessmem_kernel(ist.data, ist.indptr, ist.ii, ist.jj, ist.nnz, ist.alpha_tilde, ist.inv_mass, ist.gradC, ist.tet_indices)
    A = scipy.sparse.csr_matrix((ist.data, ist.indices, ist.indptr), shape=(ist.NT, ist.NT))
    return A


# 求两个长度为4的数组的交集
@ti.func
def intersect(a, b):   
    # a,b: 4个顶点的id, e:当前ele的id
    k=0 # 第几个共享的顶点， 0, 1, 2, 3
    c = ti.Vector([-1,-1,-1])         # 共享的顶点id存在c中
    order = ti.Vector([-1,-1,-1])     # 共享的顶点是当前ele的第几个顶点
    order2 = ti.Vector([-1,-1,-1])    # 共享的顶点是邻接ele的第几个顶点
    for i in ti.static(range(4)):     # i:当前ele的第i个顶点
        for j in ti.static(range(4)): # j:邻接ele的第j个顶点
            if a[i] == b[j]:
                c[k] = a[i]         
                order[k] = i          
                order2[k] = j
                k += 1
    return k, c, order, order2

# for cnt version, require init_A_CSR_pattern() to be called first
@ti.kernel
def fill_A_csr_lessmem_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                      indptr:ti.types.ndarray(dtype=ti.i32), 
                      ii:ti.types.ndarray(dtype=ti.i32), 
                      jj:ti.types.ndarray(dtype=ti.i32),
                      nnz:ti.i32,
                      alpha_tilde:ti.template(),
                      inv_mass:ti.template(),
                      gradC:ti.template(),
                      ele: ti.template()
                    ):
    for n in range(nnz):
        i = ii[n] # row index,  current element id
        j = jj[n] # col index,  adjacent element id, adj_id
        k = n - indptr[i] # k: 第几个非零元
        if i == j: # diag
            m1,m2,m3,m4 = inv_mass[ele[i][0]], inv_mass[ele[i][1]], inv_mass[ele[i][2]], inv_mass[ele[i][3]]
            g1,g2,g3,g4 = gradC[i,0], gradC[i,1], gradC[i,2], gradC[i,3]
            data[n] = m1*g1.norm_sqr() + m2*g2.norm_sqr() + m3*g3.norm_sqr() + m4*g4.norm_sqr() + alpha_tilde[i]
            continue
        offdiag=0.0
        n_shared_v, shared_v, shared_v_order_in_cur, shared_v_order_in_adj = intersect(ele[i], ele[j])
        for kv in range(n_shared_v): #kv 第几个共享点
            o1 = shared_v_order_in_cur[kv]
            o2 = shared_v_order_in_adj[kv]
            sv = shared_v[kv]  #sv: 共享的顶点id    shared vertex
            sm = inv_mass[sv]      #sm: 共享的顶点的质量倒数 shared inv mass
            offdiag += sm*gradC[i,o1].dot(gradC[j,o2])
        data[n] = offdiag

    
# for cnt version, require init_A_CSR_pattern() to be called first
# legacy version, now we use less memory version
# fill_A_csr_kernel(ist.data, ist.indptr, ist.ii, ist.jj, ist.nnz, ist.alpha_tilde, ist.inv_mass, ist.gradC, ist.tet_indices, ist.n_shared_v, ist.shared_v, ist.shared_v_order_in_cur, ist.shared_v_order_in_adj)
@ti.kernel
def fill_A_csr_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                      indptr:ti.types.ndarray(dtype=ti.i32), 
                      ii:ti.types.ndarray(dtype=ti.i32), 
                      jj:ti.types.ndarray(dtype=ti.i32),
                      nnz:ti.i32,
                      alpha_tilde:ti.template(),
                      inv_mass:ti.template(),
                      gradC:ti.template(),
                      ele: ti.template(),
                      n_shared_v:ti.types.ndarray(),
                      shared_v:ti.types.ndarray(),
                      shared_v_order_in_cur:ti.types.ndarray(),
                      shared_v_order_in_adj:ti.types.ndarray(),
                    ):
    for n in range(nnz):
        i = ii[n] # row index,  current element id
        j = jj[n] # col index,  adjacent element id, adj_id
        k = n - indptr[i] # k: 第几个非零元
        if i == j: # diag
            m1,m2,m3,m4 = inv_mass[ele[i][0]], inv_mass[ele[i][1]], inv_mass[ele[i][2]], inv_mass[ele[i][3]]
            g1,g2,g3,g4 = gradC[i,0], gradC[i,1], gradC[i,2], gradC[i,3]
            data[n] = m1*g1.norm_sqr() + m2*g2.norm_sqr() + m3*g3.norm_sqr() + m4*g4.norm_sqr() + alpha_tilde[i]
            continue
        offdiag=0.0
        for kv in range(n_shared_v[i, k]): #kv 第几个共享点
            o1 = shared_v_order_in_cur[i,k,kv]
            o2 = shared_v_order_in_adj[i,k,kv]
            sv = shared_v[i,k,kv]  #sv: 共享的顶点id    shared vertex
            sm = inv_mass[sv]      #sm: 共享的顶点的质量倒数 shared inv mass
            offdiag += sm*gradC[i,o1].dot(gradC[j,o2])
        data[n] = offdiag


# version 1, hand made. It is slow. By Wang Ruiqi.
# Input: .ele file
def graph_coloring_v1():
    extlib.graph_coloring.argtypes = [ctypes.c_char_p, arr_int ]
    extlib.restype = c_int
    colors = np.zeros(ist.NT, dtype=np.int32)
    abs_path = os.path.abspath(args.model_path)
    abs_path = abs_path.replace(".node", ".ele")
    model = abs_path.encode('ascii')
    tic = perf_counter()
    ncolor = extlib.graph_coloring(model, colors)
    print(f"ncolor: {ncolor}")
    print("colors of tets:",colors)
    print(f"graph_coloring_v1 time: {perf_counter()-tic:.3f}s")
    return ncolor, colors


# version 2, use pyamg. 
# Input: CSR matrix(symmetric)
# This is called in AMG_setup_phase()
def graph_coloring_v2():
    has_colored_L = [False]*num_levels
    dir = str(Path(args.model_path).parent)
    for lv in range(num_levels):
        path = dir+f'/coloring_L{lv}.txt'
        has_colored_L[lv] =  os.path.exists(path)
    has_colored = all(has_colored_L)
    if not has_colored:
        has_colored = True
    else:
        return

    from pyamg.graph import vertex_coloring
    tic = perf_counter()
    for i in range(num_levels):
        print(f"level {i}")
        Ai = fetch_A_from_cuda(i)
        colors = vertex_coloring(Ai)
        ncolor = np.max(colors)+1
        print(f"ncolor: {ncolor}")
        print("colors:",colors)
        np.savetxt(dir + f"/color_L{i}.txt", colors, fmt="%d")
        graph_coloring_to_cuda(ncolor, colors, i)
    print(f"graph_coloring_v2 time: {perf_counter()-tic:.3f}s")
    return ncolor, colors


# version 3, use newtworkx. 
# Input: CSR matrix(symmetric)
# This is called in AMG_setup_phase()
def graph_coloring_v3(A):
    import networkx as nx
    tic = perf_counter()
    net = nx.from_scipy_sparse_array(A)
    colors = nx.coloring.greedy_color(net)
    # change colors from dict to numpy array
    colors = np.array([colors[i] for i in range(len(colors))])
    ncolor = np.max(colors)+1
    print(f"ncolor: {ncolor}")
    print("colors:",colors)
    print(f"graph_coloring_v3 time: {perf_counter()-tic:.3f}s")
    return ncolor, colors


# read the color.txt
# Input: color.txt file path
def graph_coloring_read():
    model_dir = Path(args.model_path).parent
    path = model_dir / "color.txt"
    tic = perf_counter()

    require_process = True
    if require_process: #ECL_GC, # color.txt is nx3, left is node index, right is color
        colors_raw = np.loadtxt(path, dtype=np.int32, skiprows=1)
        # colors = colors_raw[:,0:2] # get first and third column
        # sort by node index
        sorted_indices = np.argsort(colors_raw[:, 0])
        sorted_colors = colors_raw[sorted_indices]
        colors = sorted_colors[:, 2]
    else: # ruiqi, no need to process
        colors = np.loadtxt(path, dtype=np.int32)

    ncolor = np.max(colors)+1
    print(f"ncolor: {ncolor}")
    print("colors:",colors)
    print(f"graph_coloring_read time: {perf_counter()-tic:.3f}s")


    graph_coloring_to_cuda(ncolor, colors,0)

    return ncolor, colors


def graph_coloring_to_cuda(ncolor, colors, lv):
    colors = np.ascontiguousarray(colors)
    extlib.fastmg_set_colors.argtypes = [arr_int, c_int, c_int, c_int]
    extlib.fastmg_set_colors(colors, colors.shape[0], ncolor, lv)


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    tic = perf_counter()
    global out_dir, ist, t_export_total, t_export, logger2, out_dir_name
    if args.auto_another_outdir:
        out_dir = use_another_outdir(out_dir)
    make_and_clean_dirs(out_dir)

    out_dir_name = str(Path(out_dir).name) 

    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=out_dir + f'/{out_dir_name}.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logger2 = logging.getLogger('logger2')
    logger2.addHandler(logging.FileHandler(out_dir + f'/build_P_time.log', 'a'))

    start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.info(start_date)
    logging.info(args)

    ist = SoftBody(args.model_path)
    ist.initialize()
    
    if args.export_mesh:
        write_mesh(out_dir + f"/mesh/{meta.frame:04d}", ist.pos.to_numpy(), ist.model_tri)

    if args.solver_type != "XPBD":
        init_direct_fill_A(ist)

    if use_graph_coloring:
        ...
        # graph_coloring_v1()
        # graph_coloring_read()
        # graph_coloring_v2()

    print(f"initialize time:", perf_counter()-tic)
    initial_frame = meta.frame
    t_export_total = 0.0
    
    timer_all = perf_counter()
    step_pbar = tqdm.tqdm(total=args.end_frame, initial=initial_frame)
    try:
        while True:
            info("\n\n----------------------")
            info(f"frame {meta.frame}")
            t = perf_counter()
            t_export = 0.0

            if args.solver_type == "XPBD":
                substep_xpbd(ist)
            else:
                substep_all_solver(ist)
            meta.frame += 1

            if args.export_mesh:
                tic = perf_counter()
                write_mesh(out_dir + f"/mesh/{meta.frame:04d}", ist.pos.to_numpy(), ist.model_tri)
                t_export += perf_counter() - tic

            t_export_total += t_export

            info(f"step time: {perf_counter() - t:.2f} s")
            step_pbar.update(1)
                
            if meta.frame >= args.end_frame:
                print("Normallly end.")
                ending(timer_all, start_date, initial_frame)
                exit()
    except KeyboardInterrupt:
        ending(timer_all, start_date, initial_frame)
        exit()
    except Exception as e:
        if args.solver_type == "AMGX":
            amgxsolver.finalize()
        logging.exception(f"Exception occurred:\n{e} ")
        raise e

if __name__ == "__main__":
    main()
