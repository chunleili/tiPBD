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

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("-max_frame", type=int, default=-1)
parser.add_argument("-max_iter", type=int, default=30)
parser.add_argument("-omega", type=float, default=0.1)
parser.add_argument("-mu", type=float, default=1e6)
parser.add_argument("-dt", type=float, default=3e-3)
parser.add_argument("-damping_coeff", type=float, default=1.0)
parser.add_argument("-gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("-total_mass", type=float, default=16000.0)
parser.add_argument("-solver_type", type=str, default="AMG", choices=["Jacobi", "GaussSeidel", "Direct", "SOR", "AMG", "HPBD"])
parser.add_argument("-model_path", type=str, default=f"data/model/bunnyBig/bunnyBig.node")# "cube" "bunny1k2k" "toy"
parser.add_argument("-kmeans_k", type=int, default=1000)
parser.add_argument("-end_frame", type=int, default=30)
parser.add_argument("-out_dir", type=str, default="result/latest/")
parser.add_argument("-export_matrix", type=int, default=False)

args = parser.parse_args()

out_dir = args.out_dir
Path(out_dir).mkdir(parents=True, exist_ok=True)
export_matrix = args.out_dir
export_mesh = True
export_residual = True
early_stop = True
export_log = True
smoother = "sor"

t_export_matrix = 0.0
t_calc_residual = 0.0
t_export_residual = 0.0
t_export_mesh = 0.0
t_save_state = 0.0


ti.init(arch=ti.cpu)


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
        self.h = self.args.dt  # time step size, default 3e-3
        self.inv_h2 = 1.0 / self.h / self.h
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


class SoftBody:
    def __init__(self, path):
        self.model_pos, self.model_tet, self.model_tri = read_tet(path, build_face_flag=True)
        self.NV = len(self.model_pos)
        self.NT = len(self.model_tet)
        self.NF = len(self.model_tri)
        self.display_indices = ti.field(ti.i32, self.NF * 3)
        self.display_indices.from_numpy(self.model_tri.flatten())

        self.name = "fine"
        if "coarse" in path:
            self.name = "coarse"
        elif "fine" in path:
            self.name = "fine"

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
        self.negative_C_minus_alpha_lambda = ti.field(ti.f32, shape=self.NT)
        self.tet_centroid = ti.Vector.field(3, ti.f32, shape=self.NT)
        self.potential_energy = ti.field(ti.f32, shape=())
        self.inertial_energy = ti.field(ti.f32, shape=())

        self.state = [
            self.pos,
        ]

        # for sparse matrix
        self.M = self.NT
        self.N = self.NV
        self.A = ti.linalg.SparseMatrix(self.M, self.M)

        info(f"Creating {self.name} instance")

    def initialize(self, reinit_style="enlarge"):
        info(f"Initializing {self.name} mesh")

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

        # reinit pos
        if reinit_style == "random":
            # random init
            random_val = np.random.rand(self.pos.shape[0], 3)
            self.pos.from_numpy(random_val)
        elif reinit_style == "enlarge":
            # init by enlarge 1.5x
            self.pos.from_numpy(self.model_pos * 1.5)

        # # set max_iter
        # if self.name == "coarse":
        #     self.max_iter = meta.args.coarse_iterations
        # elif self.name == "fine":
        #     self.max_iter = meta.args.fine_iterations
        # info(f"{self.name} max_iter:{self.max_iter}")


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
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        mass_density = meta.total_mass / total_volume
        tet_mass = mass_density * rest_volume[i]
        avg_mass = tet_mass / 4.0
        mass[ia] += avg_mass
        mass[ib] += avg_mass
        mass[ic] += avg_mass
        mass[id] += avg_mass
    for i in inv_mass:
        inv_mass[i] = 1.0 / mass[i]

    # init alpha_tilde
    # avg_alpha_tilde = 0.0
    # max_alpha_tilde = 0.0
    # min_alpha_tilde = 1e10
    for i in alpha_tilde:
        alpha_tilde[i] = meta.inv_h2 * meta.inv_mu * inv_V[i]
        # avg_alpha_tilde += alpha_tilde[i]
        # max_alpha_tilde = ti.math.max(max_alpha_tilde, alpha_tilde[i])
        # min_alpha_tilde = ti.math.min(min_alpha_tilde, alpha_tilde[i])
    # avg_alpha_tilde /= alpha_tilde.shape[0]
    # print("avg_alpha_tilde: ", avg_alpha_tilde)
    # print("max_alpha_tilde: ", max_alpha_tilde)
    # print("min_alpha_tilde: ", min_alpha_tilde)

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
    h: ti.f32,
    pos: ti.template(),
    predict_pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    damping_coeff: ti.f32,
):
    for i in pos:
        vel[i] += h * meta.gravity
        vel[i] *= damping_coeff
        old_pos[i] = pos[i]
        pos[i] += h * vel[i]
        predict_pos[i] = pos[i]


@ti.kernel
def update_velocity(h: ti.f32, pos: ti.template(), old_pos: ti.template(), vel: ti.template()):
    for i in pos:
        vel[i] = (pos[i] - old_pos[i]) / h


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
        denorminator = (
            inv_mass[p0] * g0.norm_sqr()
            + inv_mass[p1] * g1.norm_sqr()
            + inv_mass[p2] * g2.norm_sqr()
            + inv_mass[p3] * g3.norm_sqr()
        )
        residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t])
        dlambda[t] = residual[t] / (denorminator + alpha_tilde[t])
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
        g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
        gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0, g1, g2, g3


def compute_negative_C_minus_alpha_lambda(constraint, alpha_tilde, lagrangian, negative_C_minus_alpha_lambda):
    compute_negative_C_minus_alpha_lambda_kernel(constraint, alpha_tilde, lagrangian, negative_C_minus_alpha_lambda)
    return negative_C_minus_alpha_lambda.to_numpy()


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
        if pos[i][1] < -1.3:
            pos[i][1] = -1.3



def transfer_back_to_pos_mfree(x,dLambda,dpos,pos):
    dLambda.from_numpy(x)
    reset_dpos(dpos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(inv_mass, dpos, pos)
    collision(pos)

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


Residual = namedtuple('residual', ['sys', 'primary', 'dual', 'obj', 'amg', 'gs','iters','t'])

def substep_all_solver(ist, max_iter=1, solver_type="GaussSeidel", P=None, R=None):
    global t_export_matrix, t_calc_residual, t_export_residual, t_save_state
    # ist is instance of fine or coarse
    semi_euler(meta.h, ist.pos, ist.predict_pos, ist.old_pos, ist.vel, meta.damping_coeff)
    reset_lagrangian(ist.lagrangian)

    # fill M_inv and ALPHA
    inv_mass_np = ist.inv_mass.to_numpy()
    inv_mass_np = np.repeat(inv_mass_np, 3, axis=0)
    M_inv = scipy.sparse.diags(inv_mass_np)

    alpha_tilde_np = ist.alpha_tilde.to_numpy()
    ALPHA = scipy.sparse.diags(alpha_tilde_np)

    r=[]
    tol_sim = 1e-6
    for meta.ite in range(max_iter):
        t_iter_start = perf_counter()
        # ----------------------------- prepare matrices ----------------------------- #
        ist.pos_mid.from_numpy(ist.pos.to_numpy())

        compute_C_and_gradC_kernel(ist.pos_mid, ist.tet_indices, ist.B, ist.constraint, ist.gradC)
        ii, jj, vv = np.zeros(ist.NT*12, dtype=np.int32), np.zeros(ist.NT*12, dtype=np.int32), np.zeros(ist.NT*12, dtype=np.float32)
        fill_gradC_triplets_kernel(ii,jj,vv, ist.gradC, ist.tet_indices)
        G = scipy.sparse.coo_array((vv, (ii, jj)))

        # assemble A and b
        A = G @ M_inv @ G.transpose() + ALPHA
        A = scipy.sparse.csr_matrix(A, dtype=np.float64)
        b = -ist.constraint.to_numpy() - ist.alpha_tilde.to_numpy() * ist.lagrangian.to_numpy()

        if export_matrix and meta.ite == 0:
            tic = time.perf_counter()
            export_A_b(A,b,postfix=f"F{meta.frame}-{meta.ite}")
            t_export_matrix = time.perf_counter()-tic

        # -------------------------------- solve Ax=b -------------------------------- #
        x0 = np.zeros_like(b)
        rsys0 = np.linalg.norm(b-A @ x0)

        if solver_type == "GaussSeidel":
            x = np.zeros_like(b)
            for _ in range(1):
                amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
                r_norm = np.linalg.norm(A @ x - b)
                print(f"{meta.ite} r:{r_norm:.2g}")
        elif solver_type == "Direct":
            x = scipy.sparse.linalg.spsolve(A, b)
        # elif solver_type == "AMG":
        #     ramg = []
        #     x = solve_amg_SA(A, b, x0, ramg)
        #     r_Axb = ramg
        #     rgs = [None]
        elif solver_type == "AMG":
            tic = time.perf_counter()
            levels = setup_AMG(A,meta.ite)
            ramg=[]
            x0 = np.zeros_like(b,dtype=np.float64)
            tic2 = time.perf_counter()
            global update_coarse_solver
            update_coarse_solver = True
            x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=100, tol=1e-6)
            toc2 = time.perf_counter()
            logging.info(f"amg_cg_solve time {toc2-tic2}")
            rgs=[None,None]
            ramg = residuals
            r_Axb = ramg

        dlambda = x
        ist.lagrangian.from_numpy(ist.lagrangian.to_numpy() + dlambda)
        dpos = M_inv @ G.transpose() @ dlambda
        ist.pos.from_numpy(ist.pos_mid.to_numpy() + dpos.reshape(-1, 3))

        rsys2 = np.linalg.norm(b - A @ x)
        

        if export_residual:
            t_iter = time.perf_counter()-t_iter_start
            t_calc_residual_start = time.perf_counter()
            calc_dual_residual(ist.alpha_tilde, ist.lagrangian, ist.constraint, ist.dual_residual)
            # if use_primary_residual:
            primary_residual = calc_primary_residual(G, M_inv, ist.predict_pos, ist.pos, ist.lagrangian)
            primary_r = np.linalg.norm(primary_residual).astype(float)
            # else: primary_r = 0.0
            dual_r = np.linalg.norm(ist.dual_residual.to_numpy()).astype(float)
            compute_potential_energy(ist.potential_energy, ist.alpha_tilde, ist.lagrangian)
            compute_inertial_energy(ist.inertial_energy, ist.inv_mass, ist.pos, ist.predict_pos, meta.h)
            robj = (ist.potential_energy[None]+ist.inertial_energy[None])
            if export_log:
                logging.info(f"{meta.frame}-{meta.ite} r:{rsys0:.2e} {rsys2:.2e} primary:{primary_r:.2e} dual_r:{dual_r:.2e} object:{robj:.2e} iter:{len(r_Axb)} t:{t_iter:.2f}s")
            r.append(Residual([rsys0,rsys2], primary_r, dual_r, robj, ramg, rgs, len(r_Axb), t_iter))
            t_calc_residual += time.perf_counter()-t_calc_residual_start

        x_prev = x.copy()
        # gradC_prev = gradC.to_numpy().copy()

        if early_stop and meta.ite>0:
            # if rsys0 < tol_sim:
            #     break
            if r[-1].dual/r[0].dual < 0.1:
                break

    if export_residual:
        tic = time.perf_counter()
        serialized_r = [r[i]._asdict() for i in range(len(r))]
        r_json = json.dumps(serialized_r,   default=to_serializable)
        with open(out_dir+'/r/'+ f'{meta.frame}.json', 'w') as file:
            file.write(r_json)
        t_export_residual = time.perf_counter()-tic

    collsion_response(ist.pos)
    update_velocity(meta.h, ist.pos, ist.old_pos, ist.vel)


# ---------------------------------------------------------------------------- #
#                               PYAMG reproduced                               #
# ---------------------------------------------------------------------------- #
def solve_amg_SA(A,b,x0,residuals=[],tol_Axb=1e-6, max_iter_Axb=150):
    import pyamg
    ml5 = pyamg.smoothed_aggregation_solver(A,
        smooth=None,
        max_coarse=400,
        coarse_solver="pinv")
    x5 = ml5.solve(b, x0=x0.copy(), tol=tol_Axb, residuals=residuals, accel='cg', maxiter=max_iter_Axb, cycle="V")
    return x5



def chebyshev(A, x, b):
    polynomial(A, x, b, coefficients=chebyshev_coeff, iterations=1)


def setup_chebyshev(lvl, lower_bound=1.0/30.0, upper_bound=1.1, degree=3,
                    iterations=1):
    global chebyshev_coeff # FIXME: later we should store this in the level
    """Set up Chebyshev."""
    rho = approximate_spectral_radius(lvl.A)
    a = rho * lower_bound
    b = rho * upper_bound
    # drop the constant coefficient
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]
    chebyshev_coeff = coefficients
    return coefficients


def build_Ps(A, method='UA', B=None):
    """Build a list of prolongation matrices Ps from A """
    if method == 'UA':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None, symmetry='symmetric')
    elif method == 'SA' :
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400,symmetry='symmetric')
    elif method == 'CAMG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400,symmetry='symmetric')
    elif method == 'adaptive_SA':
        ml = pyamg.aggregation.adaptive_sa_solver(A, max_coarse=400, smooth=None, num_candidates=6)[0]
    elif method == 'nullspace':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None,symmetry='symmetric', B=B)
    else:
        raise ValueError(f"Method {method} not recognized")

    Ps = []
    for i in range(len(ml.levels)-1):
        Ps.append(ml.levels[i].P)

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

def setup_AMG(A,ite):
    global levels, Ps
    if not (((meta.frame%10==0) or (meta.frame==1)) and (ite==0)):
        levels = build_levels(A, Ps)
        return levels
    tic1 = perf_counter()
    B = calc_near_nullspace_GS(A)
    Ps = build_Ps(A, method='nullspace', B=B)
    levels = build_levels(A, Ps)
    if smoother == 'chebyshev':
        setup_chebyshev(levels[0], lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1)
    toc = perf_counter()
    print(f"AMG setup time: {toc-tic1:.4f}s")
    return levels


def calc_near_nullspace_GS(A):
    n=6
    tic = perf_counter()
    B = np.zeros((A.shape[0],n), dtype=np.float64)
    from pyamg.relaxation.relaxation import gauss_seidel
    for i in range(n):
        x = np.ones(A.shape[0], dtype=np.float64) + 1e-2*np.random.rand(A.shape[0])
        b = np.zeros(A.shape[0], dtype=np.float64) 
        gauss_seidel(A,x,b,iterations=20, sweep='forward')
        B[:,i] = x
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B


def amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    tic_amgcg = perf_counter()
    x = x0.copy()
    A = levels[0].A
    residuals = np.zeros(maxiter+1)
    t_vcycle = 0.0
    def psolve(b):
        x = x0.copy()
        V_cycle(levels, 0, x, b)
        return x
    bnrm2 = np.linalg.norm(b)
    atol = tol * bnrm2
    r = b - A@(x)
    rho_prev, p = None, None
    normr = np.linalg.norm(r)
    residuals[0] = normr
    for iteration in range(maxiter):
        if normr < atol:  # Are we done?
            break
        tic_vcycle = perf_counter()
        z = psolve(r)
        toc_vcycle = perf_counter()
        t_vcycle += toc_vcycle - tic_vcycle
        # print(f"Once V_cycle time: {toc_vcycle - tic_vcycle:.4f}s")
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
    toc_amgcg = perf_counter()
    t_amgcg = toc_amgcg - tic_amgcg
    # print(f"Total V_cycle time in one amg_cg_solve: {t_vcycle:.4f}s")
    # print(f"Total time of amg_cg_solve: {t_amgcg:.4f}s")
    # print(f"Time of CG(exclude v-cycle): {t_amgcg - t_vcycle:.4f}s")
    return (x),  residuals  


def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag

def presmoother(A,x,b):
    from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
    if smoother == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif smoother == 'jacobi':
        jacobi(A,x,b,iterations=10)
    elif smoother == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif smoother == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif smoother == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif smoother == 'chebyshev':
        chebyshev(A,x,b)


def postsmoother(A,x,b):
    presmoother(A,x,b)


# 实现仅第一次进入coarse_solver时计算一次P, 但每个新的A都要重新计算
# https://stackoverflow.com/a/279597/19253199
def coarse_solver(A, b):
    # global update_coarse_solver
    # if not hasattr(coarse_solver, "P") or update_coarse_solver:
    #     coarse_solver.P = pinv(A.toarray())
    #     update_coarse_solver = False
    # res = np.dot(coarse_solver.P, b)
    # # res = scipy.sparse.linalg.spsolve(A, b)
    res = np.linalg.solve(A.toarray(), b)
    return res

t_smoother = 0.0

def V_cycle(levels,lvl,x,b):
    global t_smoother
    A = levels[lvl].A
    tic = perf_counter()
    presmoother(A,x,b)
    toc = perf_counter()
    t_smoother += toc-tic
    # print(f"lvl {lvl} presmoother time: {toc-tic:.4f}s")
    tic = perf_counter()
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    toc = perf_counter()
    # print(f"lvl {lvl} restriction time: {toc-tic:.4f}s")
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        tic = perf_counter()
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
        toc = perf_counter()
        # print(f"lvl {lvl} coarse_solver time: {toc-tic:.4f}s")
    else:
        V_cycle(levels, lvl+1, coarse_x, coarse_b)
    tic = perf_counter()
    x += levels[lvl].P @ coarse_x
    toc = perf_counter()
    # print(f"lvl {lvl} interpolation time: {toc-tic:.4f}s")
    tic = perf_counter()
    postsmoother(A, x, b)
    toc = perf_counter()
    t_smoother += toc-tic
    # print(f"lvl {lvl} postsmoother time: {toc-tic:.4f}s")



# def gauss_seidel(A, x, b, iterations=1):
#     if not sparse.isspmatrix_csr(A):
#         raise ValueError("A must be csr matrix!")

#     for _iter in range(iterations):
#         # forward sweep
#         print("forward sweeping")
#         for _ in range(iterations):
#             amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

#         # backward sweep
#         print("backward sweeping")
#         for _ in range(iterations):
#             amg_core_gauss_seidel(
#                 A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
#             )
#     return x


def amg_core_gauss_seidel(Ap, Aj, Ax, x, b, row_start: int, row_stop: int, row_step: int):
    for i in range(row_start, row_stop, row_step):
        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag


def amg_core_gauss_seidel_kernel(Ap: ti.types.ndarray(dtype=int),
                                 Aj: ti.types.ndarray(dtype=int),
                                 Ax: ti.types.ndarray(dtype=float),
                                 x: ti.types.ndarray(),
                                 b: ti.types.ndarray(),
                                 row_start: int,
                                 row_stop: int,
                                 row_step: int):
    if row_step < 0:
        assert "row_step must be positive"
    for i in range(row_start, row_stop):
        if i%row_step != 0:
            continue

        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag
# ---------------------------------------------------------------------------- #
#                                compute R and P                               #
# ---------------------------------------------------------------------------- #
@ti.func
def is_in_tet_func(p, p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = ti.math.inverse(A) @ b
    return ((x[0] >= 0 and x[1] >= 0 and x[2] >= 0) and x[0] + x[1] + x[2] <= 1), x


@ti.func
def tet_centroid_func(tet_indices, pos, t):
    a, b, c, d = tet_indices[t]
    p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
    p = (p0 + p1 + p2 + p3) / 4
    return p


@ti.kernel
def compute_all_centroid(pos: ti.template(), tet_indices: ti.template(), res: ti.template()):
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



@ti.kernel
def compute_R_kernel_np(
    fine_pos: ti.template(),
    fine_tet_indices: ti.template(),
    fine_centroid: ti.template(),
    coarse_pos: ti.template(),
    coarse_tet_indices: ti.template(),
    coarse_centroid: ti.template(),
    R: ti.types.ndarray(),
):
    for i in fine_centroid:
        p = fine_centroid[i]
        flag = False
        for tc in range(coarse_tet_indices.shape[0]):
            a, b, c, d = coarse_tet_indices[tc]
            p0, p1, p2, p3 = coarse_pos[a], coarse_pos[b], coarse_pos[c], coarse_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
            if flag:
                R[tc, i] = 1
                break
        if not flag:
            print("Warning: fine tet centroid {i} not in any coarse tet")


@ti.kernel
def compute_R_based_on_kmeans_label(
    labels: ti.types.ndarray(dtype=int),
    R: ti.types.ndarray(),
):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if labels[j] == i:
                R[i, j] = 1


def compute_R_and_P(coarse, fine):
    # 计算所有四面体的质心
    print(">>Computing all tet centroid...")
    compute_all_centroid(fine.pos, fine.tet_indices, fine.tet_centroid)
    compute_all_centroid(coarse.pos, coarse.tet_indices, coarse.tet_centroid)

    # 计算R 和 P
    print(">>Computing P and R...")
    t = perf_counter()
    M, N = coarse.tet_indices.shape[0], fine.tet_indices.shape[0]
    R = np.zeros((M, N))
    compute_R_kernel_np(
        fine.pos, fine.tet_indices, fine.tet_centroid, coarse.pos, coarse.tet_indices, coarse.tet_centroid, R
    )
    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()
    print(f"Computing P and R done, time = {perf_counter() - t}")
    # print(f"writing P and R...")
    # R.mmwrite("R.mtx")
    # P.mmwrite("P.mtx")
    return R, P


def compute_R_and_P_kmeans(ist):
    print(">>Computing P and R...")
    t = perf_counter()

    from scipy.cluster.vq import vq, kmeans, whiten

    # 计算所有四面体的质心
    print(">>Computing all tet centroid...")
    compute_all_centroid(ist.pos, ist.tet_indices, ist.tet_centroid)

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = ist.tet_centroid.to_numpy()

    np.savetxt("tet_centroid.txt", input)

    N = input.shape[0]
    k = int(N / 100)
    print("N: ", N)
    print("k: ", k)

    # run kmeans
    input = whiten(input)
    print("whiten done")

    print("computing kmeans...")
    kmeans_centroids, distortion = kmeans(obs=input, k_or_guess=k, iter=20)
    labels, _ = vq(input, kmeans_centroids)

    print("distortion: ", distortion)
    print("kmeans done")

    # ----------------------------------- R and P --------------------------------- #
    # 计算R 和 P
    R = np.zeros((k, N), dtype=np.float32)

    # TODO
    compute_R_based_on_kmeans_label(labels, R)

    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()
    print(f"Computing P and R done, time = {perf_counter() - t}")

    print(f"writing P and R...")
    scipy.io.mmwrite("R.mtx", R)
    scipy.io.mmwrite("P.mtx", P)
    print(f"writing P and R done")

    return R, P

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
    if binary:
        scipy.sparse.save_npz(dir + f"A_{postfix}.npz", A)
        np.save(dir + f"b_{postfix}.npy", b)
        # A = scipy.sparse.load_npz("A.npz") # load
    else:
        scipy.io.mmwrite(dir + f"A_{postfix}.mtx", A, symmetry='symmetric')
        np.savetxt(dir + f"b_{postfix}.txt", b)

# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    make_and_clean_dirs(out_dir)

    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=out_dir + f'/latest.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.info(date)

    ist = SoftBody(meta.args.model_path)
    ist.initialize()


    while True:
        info("\n\n----------------------")
        info(f"frame {meta.frame}")
        t = perf_counter()

        substep_all_solver(ist, meta.args.max_iter, meta.args.solver_type)

        if export_mesh:
            write_mesh(out_dir + f"/mesh/{meta.frame:04d}", ist.pos.to_numpy(), ist.model_tri)
        
        meta.frame += 1
        info(f"step time: {perf_counter() - t:.2f} s")
            
        if meta.frame == meta.args.end_frame:
            exit()

if __name__ == "__main__":
    main()
