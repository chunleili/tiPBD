import taichi as ti
from taichi.lang.ops import sqrt
import numpy as np
import logging
from logging import info, warning
import scipy
import scipy.sparse as sparse
import sys, os, argparse
from time import time
from pathlib import Path
import meshio
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_at", type=int, default=-1)
parser.add_argument("-s", "--save_at", type=int, default=-1)
parser.add_argument("-m", "--max_frame", type=int, default=-1)
parser.add_argument("-e", "--log_energy_range", nargs=2, type=int, default=(-1, -1))
parser.add_argument("-r", "--log_residual_range", nargs=2, type=int, default=(-1, -1))
parser.add_argument("-p", "--pause_at", type=int, default=-1)
parser.add_argument("-c", "--coarse_iterations", type=int, default=5)
parser.add_argument("-f", "--fine_iterations", type=int, default=5)
parser.add_argument("--omega", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=1e6)
parser.add_argument("--dt", type=float, default=3e-3)
parser.add_argument("--damping_coeff", type=float, default=1.0)
parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--total_mass", type=float, default=16000.0)
parser.add_argument(
    "--solver", type=str, default="Jacobi", choices=["Jacobi", "GaussSeidel", "DirectSolver", "SOR", "AMG", "HPBD"]
)

# default_model = "bunny1k2k" # "cube" "bunny1k2k" "toy"
# parser.add_argument("--coarse_model_path", type=str, default=f"data/model/{default_model}/coarse.node")
# parser.add_argument("--fine_model_path", type=str, default=f"data/model/{default_model}/fine.node")
parser.add_argument("--model_path", type=str, default=f"data/model/bunny_small/bunny_small.node")
parser.add_argument("--kmeans_k", type=int, default=1000)

export_obj = True
proj_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = proj_dir_path + "/result/test/"
stop_frame = 10
export_matrix = True


ti.init(arch=ti.cpu)


class Meta:
    def __init__(self) -> None:
        # control parameters
        self.args = parser.parse_args()
        self.frame = 0
        self.max_frame = self.args.max_frame
        self.log_energy_range = range(*self.args.log_energy_range)
        self.log_residual_range = range(*self.args.log_residual_range)
        self.frame_to_save = self.args.save_at
        self.load_at = self.args.load_at
        self.pause = False
        self.pause_at = self.args.pause_at
        self.use_multigrid = True

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
        '*.ply'
    ]:
        files = glob.glob(os.path.join(folder_path, name))
        to_remove += (files)
    print(f"removing {len(to_remove)} files")
    for file_path in to_remove:
        os.remove(file_path)
    print(f"clean {folder_path} done")

def write_obj(filename, pos, tri):
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )
    mesh.write(filename)
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


class ArapMultigrid:
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
        self.dlambda = ti.field(ti.f32, shape=self.NT)
        self.negative_C_minus_alpha_lambda = ti.field(ti.f32, shape=self.NT)
        self.tet_centroid = ti.Vector.field(3, ti.f32, shape=self.NT)

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

        # set max_iter
        if self.name == "coarse":
            self.max_iter = meta.args.coarse_iterations
        elif self.name == "fine":
            self.max_iter = meta.args.fine_iterations
        info(f"{self.name} max_iter:{self.max_iter}")

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
    avg_alpha_tilde = 0.0
    max_alpha_tilde = 0.0
    min_alpha_tilde = 1e10
    for i in alpha_tilde:
        alpha_tilde[i] = meta.inv_h2 * meta.inv_mu * inv_V[i]
        avg_alpha_tilde += alpha_tilde[i]
        max_alpha_tilde = ti.math.max(max_alpha_tilde, alpha_tilde[i])
        min_alpha_tilde = ti.math.min(min_alpha_tilde, alpha_tilde[i])
    avg_alpha_tilde /= alpha_tilde.shape[0]
    print("avg_alpha_tilde: ", avg_alpha_tilde)
    print("max_alpha_tilde: ", max_alpha_tilde)
    print("min_alpha_tilde: ", min_alpha_tilde)

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


def substep_all_solver(ist, max_iter=1, solver="GaussSeidel", P=None, R=None):
    # ist is instance of fine or coarse
    semi_euler(meta.h, ist.pos, ist.predict_pos, ist.old_pos, ist.vel, meta.damping_coeff)
    reset_lagrangian(ist.lagrangian)

    # fill M_inv and ALPHA
    inv_mass_np = ist.inv_mass.to_numpy()
    inv_mass_np = np.repeat(inv_mass_np, 3, axis=0)
    M_inv = scipy.sparse.diags(inv_mass_np)

    alpha_tilde_np = ist.alpha_tilde.to_numpy()
    ALPHA = scipy.sparse.diags(alpha_tilde_np)

    for ite in range(max_iter):
        # ----------------------------- prepare matrices ----------------------------- #
        ist.pos_mid.from_numpy(ist.pos.to_numpy())

        compute_C_and_gradC_kernel(ist.pos_mid, ist.tet_indices, ist.B, ist.constraint, ist.gradC)
        ii, jj, vv = np.zeros(ist.NT*12, dtype=np.int32), np.zeros(ist.NT*12, dtype=np.int32), np.zeros(ist.NT*12, dtype=np.float32)
        fill_gradC_triplets_kernel(ii,jj,vv, ist.gradC, ist.tet_indices)
        G = scipy.sparse.coo_array((vv, (ii, jj)))

        # assemble A and b
        A = G @ M_inv @ G.transpose() + ALPHA
        A = scipy.sparse.csr_matrix(A)
        b = -ist.constraint.to_numpy() - ist.alpha_tilde.to_numpy() * ist.lagrangian.to_numpy()

        if meta.frame == stop_frame and export_matrix:
            print(f"writing A and b to {out_dir}")
            scipy.io.mmwrite(out_dir + f"A.mtx", A)
            np.savetxt(out_dir + f"b.txt", b)
            exit()
        # -------------------------------- solve Ax=b -------------------------------- #
        x0 = np.zeros_like(b)
        A = scipy.sparse.csr_matrix(A)

        if solver == "GaussSeidel":
            x = np.zeros_like(b)
            for _ in range(1):
                amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
                r_norm = np.linalg.norm(A @ x - b)
                print(f"{ite} r:{r_norm:.2g}")
        elif solver == "DirectSolver":
            x = scipy.sparse.linalg.spsolve(A, b)
        elif solver == "AMG":
            x = solve_pyamg_my2(A, b, x0, R, P)

        dlambda = x
        ist.lagrangian.from_numpy(ist.lagrangian.to_numpy() + dlambda)
        dpos = M_inv @ G.transpose() @ dlambda
        ist.pos.from_numpy(ist.pos_mid.to_numpy() + dpos.reshape(-1, 3))

    collsion_response(ist.pos)
    update_velocity(meta.h, ist.pos, ist.old_pos, ist.vel)


# ---------------------------------------------------------------------------- #
#                               PYAMG reproduced                               #
# ---------------------------------------------------------------------------- #


def solve_pyamg_my2(A, b, x0, R, P):
    tol = 1e-3
    # residuals = r_norm_list
    residuals = []
    maxiter = 1

    A2 = R @ A @ P

    x = x0

    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0
    while True:  # it <= maxiter and normr >= tol:
        gauss_seidel(A, x, b, iterations=1)  # presmoother

        residual = b - A @ x

        coarse_b = R @ residual  # restriction

        coarse_x = np.zeros_like(coarse_b)

        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)

        x += P @ coarse_x  # coarse grid correction

        gauss_seidel(A, x, b, iterations=1)  # postsmoother

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x


def gauss_seidel(A, x, b, iterations=1):
    if not sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        # forward sweep
        print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

        # backward sweep
        print("backward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(
                A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
            )
    return x


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
    t = time()
    M, N = coarse.tet_indices.shape[0], fine.tet_indices.shape[0]
    R = np.zeros((M, N))
    compute_R_kernel_np(
        fine.pos, fine.tet_indices, fine.tet_centroid, coarse.pos, coarse.tet_indices, coarse.tet_centroid, R
    )
    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()
    print(f"Computing P and R done, time = {time() - t}")
    # print(f"writing P and R...")
    # R.mmwrite("R.mtx")
    # P.mmwrite("P.mtx")
    return R, P


def compute_R_and_P_kmeans(ist):
    print(">>Computing P and R...")
    t = time()

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
    print(f"Computing P and R done, time = {time() - t}")

    print(f"writing P and R...")
    scipy.io.mmwrite("R.mtx", R)
    scipy.io.mmwrite("P.mtx", P)
    print(f"writing P and R done")

    return R, P


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if meta.args.solver == "AMG":
        fine = ArapMultigrid(meta.args.fine_model_path)
        coarse = ArapMultigrid(meta.args.coarse_model_path)
        fine.initialize()
        coarse.initialize()
        ist = fine
    else:
        ist = ArapMultigrid(meta.args.model_path)
        ist.initialize()

    clean_result_dir(out_dir)

    while True:
        info("\n\n----------------------")
        info(f"frame {meta.frame}")
        t = time()

        substep_all_solver(ist, 30, "GaussSeidel")

        if export_obj:
            write_obj(out_dir + f"{meta.frame:04d}.obj", ist.pos.to_numpy(), ist.model_tri)
        
        meta.frame += 1
        info(f"step time: {time() - t:.2g} s")
            
        if meta.frame == meta.max_frame:
            exit()

if __name__ == "__main__":
    main()
