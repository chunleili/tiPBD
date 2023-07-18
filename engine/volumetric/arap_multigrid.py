"""
Modified YP multi-grid solver for ARAP
"""
import taichi as ti
from taichi.lang.ops import sqrt
import numpy as np
import logging
from logging import info
import scipy
import scipy.io as sio
from scipy.sparse import coo_matrix, spdiags, kron
from scipy.io import mmwrite
import sys, os, argparse
from time import time

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_at", type=int, default=-1)
parser.add_argument("-s", "--save_at", type=int, default=-1)
parser.add_argument("-m", "--max_frame", type=int, default=-1)
parser.add_argument("-e", "--log_energy_range", nargs=2, type=int, default=(-1, -1))
parser.add_argument("-r", "--log_residual_range", nargs=2, type=int, default=(-1, -1))
parser.add_argument("-p", "--pause_at", type=int, default=-1)
parser.add_argument("-c", "--coarse_iterations", type=int, default=5)
parser.add_argument("-f", "--fine_iterations", type=int, default=5)
parser.add_argument("--model", type=str, default="cube")
parser.add_argument("--omega", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=1e6)
parser.add_argument("--dt", type=float, default=3e-3)
parser.add_argument("--damping_coeff", type=float, default=1.0)
parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--total_mass", type=float, default=16000.0)
parser.add_argument("--solver_type", type=str, default="Jacobian", choices=["Jacobian", "GaussSeidel", "DirectSolver"])
parser.add_argument("--multigrid_type", type=str, default="HPBD", choices=["HPBD", "AMG", "GMG"])

ti.init(arch=ti.cpu, debug=True, kernel_profiler=True)


class Meta:
    ...


meta = Meta()

# control parameters
meta.args = parser.parse_args()
meta.frame = 0
meta.max_frame = meta.args.max_frame
meta.log_energy_range = range(*meta.args.log_energy_range)
meta.log_residual_range = range(*meta.args.log_residual_range)
meta.frame_to_save = meta.args.save_at
meta.load_at = meta.args.load_at
meta.pause = False
meta.pause_at = meta.args.pause_at
meta.use_multigrid = True


# physical parameters
meta.omega = meta.args.omega  # SOR factor, default 0.1
meta.mu = meta.args.mu  # Lame's second parameter, default 1e6
meta.inv_mu = 1.0 / meta.mu
meta.h = meta.args.dt  # time step size, default 3e-3
meta.inv_h2 = 1.0 / meta.h / meta.h
meta.gravity = ti.Vector(meta.args.gravity)  # gravity, default (0, 0, 0)
meta.damping_coeff = meta.args.damping_coeff  # damping coefficient, default 1.0
meta.total_mass = meta.args.total_mass  # total mass, default 16000.0
# meta.mass_density = 2000.0


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


class ArapMultigrid:
    def __init__(self, path):
        self.model_pos, self.model_tet, self.model_tri = read_tetgen(path)
        self.NV = len(self.model_pos)
        self.NT = len(self.model_tet)
        self.NF = len(self.model_tri)
        self.name = ""
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
        self.display_indices = ti.field(ti.i32, self.NF * 3)
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
        self.C_plus_alpha_lambda = ti.field(ti.f32, shape=self.NT)

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
        self.display_indices.from_numpy(self.model_tri.flatten())

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

        # assemble inv_mass_mat and alpha_tilde_mat
        self.inv_mass_mat = assemble_inv_mass_mat(self.inv_mass)
        self.alpha_tilde_mat = assemble_alpha_tilde_mat(self.alpha_tilde)

    def one_iter_direct_solver(self, A=None):
        if A is None:
            self.solve_constraints()
            gradC_mat = self.assemble_gradC()
            A = assemble_A(gradC_mat, self.inv_mass_mat, self.alpha_tilde_mat)

        b = self.residual

        x = solve_direct_solver(A, b)

        dpos = self.inv_mass_mat @ gradC_mat.transpose() @ x
        self.pos.from_numpy(self.pos_mid.to_numpy() + dpos.reshape(-1, 3))

        self.schur_residual = A @ x - b.to_numpy()
        print(f"schur residual for direct solver: {np.linalg.norm(self.schur_residual)}")

    def one_iter_jacobian(self):
        project_constraints(
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
            self.dpos,
        )

    def iterate_single_mesh(self, solver_type, A=None):
        reset_lagrangian(self.lagrangian)
        for ite in range(self.max_iter):
            if ite == 0:
                log_residual(meta.frame, meta.residual_filename, self)
            log_energy(meta.frame, meta.energy_filename, self)
            if solver_type == "Jacobian":
                self.one_iter_jacobian()
            if solver_type == "DirectSolver":
                self.one_iter_direct_solver(A)
            log_residual(meta.frame, meta.residual_filename, self)
        collsion_response(self.pos)
        update_velocity(meta.h, self.pos, self.old_pos, self.vel)

    def substep_single_mesh(self, solver_type):
        semi_euler(meta.h, self.pos, self.predict_pos, self.old_pos, self.vel, meta.damping_coeff)
        self.iterate_single_mesh(solver_type)
        update_velocity(meta.h, self.pos, self.old_pos, self.vel)

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

    def assemble_gradC(self):
        gradC_builder = ti.linalg.SparseMatrixBuilder(self.M, 3 * self.N, max_num_triplets=12 * self.M)
        # fill_gradC(gradC_builder, self.gradC, self.tet_indices)
        compute_gradC_kernel(self.pos_mid, self.tet_indices, self.B, self.gradC, gradC_builder)
        gradC_mat = gradC_builder.build()
        return gradC_mat

    def compute_dlambda(self):
        compute_dlambda_kernel(
            self.pos_mid, self.tet_indices, self.inv_mass, self.lagrangian, self.B, self.alpha_tilde, self.dlambda
        )
        return self.dlambda


def assemble_inv_mass_mat(inv_mass):
    N = inv_mass.shape[0]
    inv_mass_builder = ti.linalg.SparseMatrixBuilder(3 * N, 3 * N, max_num_triplets=3 * N)
    fill_invmass(inv_mass_builder, inv_mass)
    inv_mass_mat = inv_mass_builder.build()
    return inv_mass_mat


def assemble_alpha_tilde_mat(alpha_tilde):
    M = alpha_tilde.shape[0]
    alpha_tilde_builder = ti.linalg.SparseMatrixBuilder(M, M, max_num_triplets=12 * M)
    fill_diag(alpha_tilde_builder, alpha_tilde)
    alpha_tilde_mat = alpha_tilde_builder.build()
    return alpha_tilde_mat


# compute schur complement as A
def assemble_A(gradC_mat, inv_mass_mat, alpha_tilde_mat):
    A = gradC_mat @ inv_mass_mat @ gradC_mat.transpose() + alpha_tilde_mat
    return A


@ti.kernel
def fill_diag(A: ti.types.sparse_matrix_builder(), val: ti.template()):
    for i in range(val.shape[0]):
        A[i, i] += val[i]


# fill gradC
@ti.kernel
def fill_gradC(
    A: ti.types.sparse_matrix_builder(),
    gradC: ti.template(),
    tet_indices: ti.template(),
):
    for j in range(tet_indices.shape[0]):
        ind = tet_indices[j]
        for p in range(4):
            for d in range(3):
                pid = ind[p]
                A[j, 3 * pid + d] += gradC[j, p][d]


@ti.kernel
def fill_invmass(A: ti.types.sparse_matrix_builder(), val: ti.template()):
    for i in range(val.shape[0]):
        A[3 * i, 3 * i] += val[i]
        A[3 * i + 1, 3 * i + 1] += val[i]
        A[3 * i + 2, 3 * i + 2] += val[i]


def coarse_to_fine_pos(P, fine, coarse):
    fine.pos.from_numpy(P @ coarse.pos.to_numpy())


def fine_to_coarse_pos(R, fine, coarse):
    coarse.pos.from_numpy(R @ fine.pos.to_numpy())


def restriction(R, fine_val):
    return R @ fine_val


def prolongation(P, coarse_val):
    return P @ coarse_val


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
def compute_dlambda_kernel(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    inv_mass: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    alpha_tilde: ti.template(),
    dlambda: ti.template(),
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
        constraint = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
        denorminator = (
            inv_mass[p0] * g0.norm_sqr()
            + inv_mass[p1] * g1.norm_sqr()
            + inv_mass[p2] * g2.norm_sqr()
            + inv_mass[p3] * g3.norm_sqr()
        )
        dlambda[t] = -(constraint + alpha_tilde[t] * lagrangian[t]) / (denorminator + alpha_tilde[t])
        lagrangian[t] += dlambda[t]


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


@ti.kernel
def compute_gradC_kernel(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    gradC: ti.template(),
    gradC_mat: ti.types.sparse_matrix_builder(),
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
        gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = compute_gradient(U, S, V, B[t])

        ind = tet_indices[t]
        for p in range(4):
            for d in range(3):
                pid = ind[p]
                gradC_mat[t, 3 * pid + d] += gradC[t, p][d]


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


@ti.kernel
def compute_inertial(mass: ti.template(), pos: ti.template(), predict_pos: ti.template()) -> ti.f32:
    it = 0.0
    for i in pos:
        it += mass[i] * (pos[i] - predict_pos[i]).norm_sqr()
    return it * 0.5


@ti.kernel
def compute_potential_energy(
    pos: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    alpha_tilde: ti.template(),
) -> ti.f32:
    pe = 0.0
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id]
        D_s = ti.Matrix.cols([b - a, c - a, d - a])
        F = D_s @ B[i]
        U, S, V = ti.svd(F)
        if S[2, 2] < 0.0:  # S[2, 2] is the smallest singular value
            S[2, 2] *= -1.0
        constraint_squared = (S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2
        pe += (1.0 / alpha_tilde[i]) * constraint_squared
    return pe * 0.5


def compute_energy(mass, pos, predict_pos, tet_indices, B, alpha_tilde):
    it = compute_inertial(mass, pos, predict_pos)
    pe = compute_potential_energy(pos, tet_indices, B, alpha_tilde)
    return it + pe, it, pe


def log_energy(frame, filename_to_save, instance):
    if frame in meta.log_energy_range:
        te, it, pe = compute_energy(
            instance.mass, instance.pos, instance.predict_pos, instance.tet_indices, instance.B, instance.alpha_tilde
        )
        info(f"energy:\t{te}")
        with open(filename_to_save, "a") as f:
            np.savetxt(f, np.array([te]), fmt="%.4e", delimiter="\t")


def log_residual(frame, filename_to_save, instance):
    if frame in meta.log_residual_range:
        r_norm = np.linalg.norm(instance.residual.to_numpy())
        logging.info("residual:\t{}".format(r_norm))
        with open(filename_to_save, "a") as f:
            np.savetxt(f, np.array([r_norm]), fmt="%.4e", delimiter="\t")


def save_state(filename, fine, coarse):
    state = fine.state + coarse.state
    state.insert(0, meta.frame)
    for i in range(1, len(state)):
        state[i] = state[i].to_numpy()
    np.savez(filename, *state)
    logging.info(f"saved state to '{filename}', totally saved {len(state)} variables")


def load_state(filename, fine, coarse):
    npzfile = np.load(filename)
    state = fine.state + coarse.state
    meta.frame = int(npzfile["arr_0"])
    for i in range(1, len(state)):
        state[i].from_numpy(npzfile["arr_" + str(i)])

    logging.info(f"loaded state from '{filename}', totally loaded {len(state)} variables")


def substep_hpbd(P, R, fine, coarse, solver_type="Jacobian"):
    semi_euler(meta.h, fine.pos, fine.predict_pos, fine.old_pos, fine.vel, meta.damping_coeff)
    fine_to_coarse_pos(R, fine, coarse)
    coarse.iterate_single_mesh(solver_type)
    coarse_to_fine_pos(P, fine, coarse)
    fine.iterate_single_mesh(solver_type)
    update_velocity(meta.h, fine.pos, fine.old_pos, fine.vel)


# ---------------------------------------------------------------------------- #
#                                      AMG                                     #
# ---------------------------------------------------------------------------- #
def solve_direct_solver(A, b):
    t = time()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)
    print(f"direct solver time: {time() - t}")
    print(f"shape of A: {A.shape}")
    print(f"solve success: {solver.info()}")
    return x


@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    for i in src:
        dst[i] = src[i]


def update_pos_from_dlambda(gradC_mat, dlambda, pos_mid, inv_mass_mat, pos):
    dpos = inv_mass_mat @ gradC_mat.transpose() @ dlambda.to_numpy()
    pos.from_numpy(pos_mid.to_numpy() + dpos.reshape(-1, 3))


def substep_amg(P, R, fine, coarse, solver_type):
    semi_euler(meta.h, fine.pos, fine.predict_pos, fine.old_pos, fine.vel, meta.damping_coeff)

    # pre-smooth jacobian:
    reset_lagrangian(fine.lagrangian)
    for ite in range(3):
        fine.one_iter_jacobian()

    # assemble A1, b1, x1, r1
    gradC_mat = fine.assemble_gradC()
    A1 = assemble_A(gradC_mat, fine.inv_mass_mat, fine.alpha_tilde_mat)
    b1 = fine.residual.to_numpy()
    x1 = fine.dlambda.to_numpy()
    r1 = b1 - A1 @ x1

    # restriction: pass r1 to r2 and construct A2
    r2 = R @ r1
    A2 = R @ A1 @ P

    # solve coarse level A2E2=r2
    E2 = solve_direct_solver(A2, r2)

    # prolongation:
    E1 = P @ E2
    x1 += E1
    fine.dlambda.from_numpy(x1)

    dpos = fine.inv_mass_mat @ gradC_mat.transpose() @ fine.dlambda
    fine.pos.from_numpy(fine.pos_mid.to_numpy() + dpos.reshape(-1, 3))

    # post-smooth jacobian:
    reset_lagrangian(fine.lagrangian)
    for ite in range(3):
        fine.one_iter_jacobian()
    x1 = fine.dlambda.to_numpy()

    r1_new = b1 - A1 @ x1
    print(f"ite: {ite}, b1:{np.linalg.norm(b1)}, r1: {np.linalg.norm(r1)}, r1_new: {np.linalg.norm(r1_new)}")

    collsion_response(fine.pos)
    update_velocity(meta.h, fine.pos, fine.old_pos, fine.vel)


# ---------------------------------------------------------------------------- #
#                                compute R and P                               #
# ---------------------------------------------------------------------------- #
@ti.func
def is_in_tet_func(p, p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = ti.math.inverse(A) @ b
    return ((x[0] >= 0 and x[1] >= 0 and x[2] >= 0) and x[0] + x[1] + x[2] <= 1), x


@ti.kernel
def compute_R_kernel(
    fine_pos: ti.template(),
    p_to_tet: ti.template(),
    coarse_pos: ti.template(),
    coarse_tet_indices: ti.template(),
    R: ti.types.sparse_matrix_builder(),
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
                R[tc, tf] += 1
                break
        if not flag:
            print(f"WARNING: point {i} not in any tet")
            min_dist = 1e10
            min_indx = -1
            for ic in range(coarse_pos.shape[0]):
                dist = (p - coarse_pos[ic]).norm()
                if dist < min_dist:
                    min_dist = dist
                    min_indx = ic
            tc = p_to_tet[min_indx]
            print(
                f"fine point {i}({p}) closest to coarse point {min_indx}({coarse_pos[min_indx]}), which is in tet {tc}, min_dist = {min_dist}"
            )
            R[tc, tf] += 1


def compute_R_and_P(coarse, fine):
    print("Computing P and R...")
    t = time()
    M, N = coarse.tet_indices.shape[0], fine.tet_indices.shape[0]
    R_builder = ti.linalg.SparseMatrixBuilder(M, N, max_num_triplets=40 * M)
    compute_R_kernel(fine.pos, fine.par_2_tet, coarse.pos, coarse.tet_indices, R_builder)
    R = R_builder.build()
    P = R.transpose()
    print(f"Computing P and R done, time = {time() - t}")
    return R, P


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_path = ""
    if meta.args.model == "bunny":
        model_path = "data/model/bunny1k2k/"
    elif meta.args.model == "cube":
        model_path = "data/model/cube/"
    fine_model_path = model_path + "fine"
    coarse_model_path = model_path + "coarse"

    fine = ArapMultigrid(fine_model_path)
    coarse = ArapMultigrid(coarse_model_path)

    fine.initialize()
    coarse.initialize()

    if coarse.max_iter == 0:
        suffix = "onlyfine"
        info("#############################################")
        info("########## Using Only Fine Mesh ###########")
        info("#############################################")
        P = sio.mmread(model_path + "P.mtx")
        R = sio.mmread(model_path + "R.mtx")
    elif fine.max_iter == 0:
        suffix = "onlycoarse"
        info("#############################################")
        info("########## Using Only Coarse Mesh ###########")
        info("#############################################")
        P = sio.mmread(model_path + "P.mtx")
        R = sio.mmread(model_path + "R.mtx")
    elif fine.max_iter != 0 and coarse.max_iter != 0:
        if meta.args.multigrid_type == "HPBD":
            info("#############################################")
            info("########## Using HPBD Multigrid #############")
            info("#############################################")
            suffix = "hpbd"
            P = sio.mmread(model_path + "P.mtx")
            R = sio.mmread(model_path + "R.mtx")
        elif meta.args.multigrid_type == "AMG":
            info("#############################################")
            info("########## Using AMG Multigrid ##############")
            info("#############################################")
            suffix = "amg"
            R, P = compute_R_and_P(coarse, fine)
        elif meta.args.multigrid_type == "GMG":
            info("#############################################")
            info("########## Using GMG Multigrid ##############")
            info("#############################################")
            suffix = "gmg"

    if meta.args.solver_type == "Jacobian":
        info("#############################################")
        info("########## Using Jacobian Solver ############")
        info("#############################################")
    elif meta.args.solver_type == "DirectSolver":
        info("#############################################")
        info("########## Using Direct Solver ##############")
        info("#############################################")

    meta.energy_filename = "result/log/totalEnergy_" + (suffix) + ".txt"
    meta.residual_filename = "result/log/residual_" + (suffix) + ".txt"
    if os.path.exists(meta.energy_filename):
        os.remove(meta.energy_filename)
    if os.path.exists(meta.residual_filename):
        os.remove(meta.residual_filename)

    save_state_filename = "result/save/frame_"
    if meta.load_at != -1:
        meta.filename_to_load = save_state_filename + str(meta.load_at) + ".npz"
        load_state(meta.filename_to_load, fine, coarse)

    window = ti.ui.Window("3D ARAP FEM XPBD", (1300, 900), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, 10)
    camera.lookat(0, 0, 0)
    camera.fov(45)
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
    gui = window.get_gui()
    wire_frame = True
    show_coarse_mesh = True
    show_fine_mesh = True
    while window.running:
        scene.ambient_light((0.8, 0.8, 0.8))
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        if window.is_pressed(ti.ui.ESCAPE):
            window.running = False

        if meta.frame == meta.pause_at:
            meta.pause = True
        if window.is_pressed(ti.ui.SPACE):
            meta.pause = not meta.pause

        gui.text("frame {}".format(meta.frame))
        meta.pause = gui.checkbox("pause", meta.pause)
        wire_frame = gui.checkbox("wireframe", wire_frame)
        show_coarse_mesh = gui.checkbox("show coarse mesh", show_coarse_mesh)
        show_fine_mesh = gui.checkbox("show fine mesh", show_fine_mesh)
        coarse.max_iter = gui.slider_int("coarse_iter", coarse.max_iter, 0, 50)
        fine.max_iter = gui.slider_int("fine_iter", fine.max_iter, 0, 50)
        gui.text("total iterations: {}".format(coarse.max_iter + fine.max_iter))

        if meta.frame == meta.frame_to_save:
            save_state(save_state_filename + str(meta.frame), fine, coarse)

        if not meta.pause:
            info("----")
            info(f"frame {meta.frame}")
            t = time()
            if coarse.max_iter == 0 and fine.max_iter != 0:
                fine.substep_single_mesh(meta.args.solver_type)
                show_coarse_mesh = False
            elif fine.max_iter == 0 and coarse.max_iter != 0:
                coarse.substep_single_mesh(meta.args.solver_type)
                show_fine_mesh = False
            elif fine.max_iter != 0 and coarse.max_iter != 0:
                if meta.args.multigrid_type == "HPBD":
                    substep_hpbd(P, R, fine, coarse, meta.args.solver_type)
                elif meta.args.multigrid_type == "AMG":
                    substep_amg(P, R, fine, coarse, meta.args.solver_type)
                elif meta.args.multigrid_type == "GMG":
                    substep_gmg(P, R, fine, coarse, meta.args.solver_type)

            # ti.profiler.print_kernel_profiler_info()
            info(f"step time: {time() - t}")

            meta.frame += 1

        if meta.frame == meta.max_frame:
            window.running = False

        if show_fine_mesh:
            scene.mesh(fine.pos, fine.display_indices, color=(1.0, 0.5, 0.5), show_wireframe=wire_frame)

        if show_coarse_mesh:
            scene.mesh(coarse.pos, coarse.display_indices, color=(0.0, 0.5, 1.0), show_wireframe=wire_frame)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
