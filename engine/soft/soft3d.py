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
import pyamg
import ctypes
import numpy.ctypeslib as ctl
import datetime
import tqdm

prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(prj_path)
from engine.solver.build_Ps import build_Ps
from engine.solver.amg_python import AMG_python
from engine.file_utils import process_dirs,  do_restart, save_state, export_A_b
from engine.mesh_io import write_mesh, read_tet
from engine.common_args import add_common_args
from engine.init_extlib import init_extlib
from engine.solver.amg_cuda import AmgCuda

parser = argparse.ArgumentParser()

parser = add_common_args(parser)

parser.add_argument("-mu", type=float, default=1e6)
parser.add_argument("-damping_coeff", type=float, default=1.0)
parser.add_argument("-total_mass", type=float, default=16000.0)
parser.add_argument("-model_path", type=str, default=f"data/model/bunny1k2k/coarse.node")
# "data/model/cube/minicube.node"
# "data/model/bunny1k2k/coarse.node"
# "data/model/bunny_small/bunny_small.node"
# "data/model/bunnyBig/bunnyBig.node"
# "data/model/bunny85w/bunny85w.node"
parser.add_argument("-reinit", type=str, default="enlarge", choices=["", "random", "enlarge"])
parser.add_argument("-large", action="store_true")
parser.add_argument("-samll", action="store_true")

parser.add_argument("-omega", type=float, default=0.1)
parser.add_argument("-smoother_type", type=str, default="jacobi")


args = parser.parse_args()

if args.large:
    args.model_path = f"data/model/bunny85w/bunny85w.node"
if args.samll:
    args.model_path = f"data/model/bunny1k2k/coarse.node"

if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)


arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
c_int = ctypes.c_int


class SoftBody:
    def __init__(self, path):
        self.frame = 0
        self.ite = 0

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

        self.n_outer_all = []
        self.t_avg_iter=[]
        self.ResidualData = namedtuple('residual', ['dual', 'ninner','t']) #residual for one outer iter
        self.all_stalled = []
        info(f"Creating instance done")

    def initialize(self):
        info(f"Initializing mesh")

        # read models
        self.model_pos = self.model_pos.astype(np.float32)
        self.model_tet = self.model_tet.astype(np.int32)
        self.pos.from_numpy(self.model_pos)
        self.tet_indices.from_numpy(self.model_tet)

        inv_mu = 1.0 / args.mu
        inv_h2 = 1.0 / args.delta_t / args.delta_t
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
            inv_mu,
            inv_h2,
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

        self.alpha_tilde_np = self.alpha_tilde.to_numpy()


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
    inv_mu: ti.f32,
    inv_h2: ti.f32,
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
    #     mass_density = args.total_mass / total_volume
    #     tet_mass = mass_density * rest_volume[i]
    #     avg_mass = tet_mass / 4.0
    #     mass[ia] += avg_mass
    #     mass[ib] += avg_mass
    #     mass[ic] += avg_mass
    #     mass[id] += avg_mass
    for i in inv_mass:
        inv_mass[i] = 1.0 

    for i in alpha_tilde:
        alpha_tilde[i] = inv_h2 * inv_mu * inv_V[i]

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
    gravity: ti.template(),
):
    for i in pos:
        vel[i] += delta_t * gravity
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
    omega: ti.f32
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
        pos[p0] += omega * inv_mass[p0] * dlambda[t] * gradC[t, 0]
        pos[p1] += omega * inv_mass[p1] * dlambda[t] * gradC[t, 1]
        pos[p2] += omega * inv_mass[p2] * dlambda[t] * gradC[t, 2]
        pos[p3] += omega * inv_mass[p3] * dlambda[t] * gradC[t, 3]
        dpos[p0] += omega * inv_mass[p0] * dlambda[t] * gradC[t, 0]
        dpos[p1] += omega * inv_mass[p1] * dlambda[t] * gradC[t, 1]
        dpos[p2] += omega * inv_mass[p2] * dlambda[t] * gradC[t, 2]
        dpos[p3] += omega * inv_mass[p3] * dlambda[t] * gradC[t, 3]


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
    omega:ti.f32
):
    for i in range(inv_mass.shape[0]):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]

def transfer_back_to_pos_mfree(x, ist):
    ist.dlambda.from_numpy(x)
    reset_dpos(ist.dpos)
    transfer_back_to_pos_mfree_kernel(ist.gradC, ist.tet_indices, ist.inv_mass, ist.dlambda, ist.lagrangian, ist.dpos)
    update_pos(ist.inv_mass, ist.dpos, ist.pos, args.omega)
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
    return ((ist.frame%args.setup_interval==0 or (args.restart==True and ist.frame==args.restart_frame)) and (ist.ite==0))



def smoother_name2type(name):
    if name == "chebyshev":
        return 1
    elif name == "jacobi":
        return 2
    elif name == "gauss_seidel":
        return 3
    else:
        raise ValueError(f"smoother name {name} not supported")



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


# def AMG_RAP():
#     tic3 = time.perf_counter()
#     # A = fill_A_csr_ti(ist)
#     # cuda_set_A0(A)
#     for lv in range(ist.num_levels-1):
#         extlib.fastmg_RAP(lv) 
#     logging.info(f"    RAP time: {(time.perf_counter()-tic3)*1000:.0f}ms")


def AMG_dlam2dpos(x):
    tic = time.perf_counter()
    transfer_back_to_pos_mfree(x, ist)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")



def do_export_r(r):
    tic = time.perf_counter()
    serialized_r = [r[i]._asdict() for i in range(len(r))]
    r_json = json.dumps(serialized_r)
    with open(args.out_dir+'/r/'+ f'{ist.frame}.json', 'w') as file:
        file.write(r_json)
    ist.t_export += time.perf_counter()-tic


def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))


def AMG_calc_r(r,fulldual0, tic_iter, r_Axb):
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
        logging.info(f"{ist.frame}-{ist.ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual0:{dual0:.2e} dual:{dual_r:.2e} iter:{len(r_Axb)}")
    r.append(ist.ResidualData(dual_r, len(r_Axb), t_iter))

    ist.t_export += perf_counter()-tic
    return dual0






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


def fastFill_set():
    extlib.fastmg_set_A0_from_fastFillSoft()


def substep_all_solver(ist):
    tic1 = time.perf_counter()
    gravity = ti.Vector(args.gravity)
    semi_euler(args.delta_t, ist.pos, ist.predict_pos, ist.old_pos, ist.vel, args.damping_coeff, gravity)
    reset_lagrangian(ist.lagrangian)
    r = [] # residual list of one frame
    logging.info(f"pre-loop time: {(perf_counter()-tic1)*1000:.0f}ms")
    for ist.ite in range(args.maxiter):
        tic_iter = perf_counter()
        ist.pos_mid.from_numpy(ist.pos.to_numpy())
        compute_C_and_gradC_kernel(ist.pos_mid, ist.tet_indices, ist.B, ist.constraint, ist.gradC)
        if ist.ite==0:
            fulldual0 = calc_dual(ist)
        b = AMG_b(ist)
        if not args.use_cuda:
            x, r_Axb = AMG_python(b,args,ist,fill_A_csr_ti,should_setup,copy_A=True)
        else:
            if args.solver_type == "AMG":
                x, r_Axb = amg_cuda.AMG_cuda(b)
            elif args.solver_type == "AMGX":
                x, r_Axb = AMG_amgx(b)
        AMG_dlam2dpos(x)
        dual0 = AMG_calc_r(r, fulldual0, tic_iter, r_Axb)
        logging.info(f"iter time(with export): {(perf_counter()-tic_iter)*1000:.0f}ms")
        if r[-1].dual<args.tol:
            break
        # if is_stall(r):
        #     logging.info("Stall detected, break")
        #     break
        if r[-1].dual / r[0].dual <args.rtol:
            break
        if is_diverge(r, r_Axb):
            logging.error("Diverge detected")
            raise ValueError("Diverge detected")
    
    tic = time.perf_counter()
    logging.info(f"n_outer: {len(r)}")
    ist.n_outer_all.append(len(r))
    if args.export_residual:
        do_export_r(r)
    collsion_response(ist.pos)
    update_vel(args.delta_t, ist.pos, ist.old_pos, ist.vel)
    logging.info(f"post-loop time: {(time.perf_counter()-tic)*1000:.0f}ms")
    ist.t_avg_iter.append((time.perf_counter()-tic1)/ist.n_outer_all[-1])
    logging.info(f"avg iter frame {ist.frame}: {ist.t_avg_iter[-1]*1000:.0f}ms")



# if in last 5 iters, residuals not change 0.1%, then it is stalled
def is_stall(r):
    if (ist.ite < 5):
        return False
    # a=np.array([r[-1].dual, r[-2].dual,r[-3].dual,r[-4].dual,r[-5].dual])
    inc1 = r[-1].dual/r[-2].dual
    inc2 = r[-2].dual/r[-3].dual
    inc3 = r[-3].dual/r[-4].dual
    inc4 = r[-4].dual/r[-5].dual
    
    # if all incs is in [0.999,1.001]
    if np.all((inc1>0.999) & (inc1<1.001) & (inc2>0.999) & (inc2<1.001) & (inc3>0.999) & (inc3<1.001) & (inc4>0.999) & (inc4<1.001)):
        logging.warning(f"Stall at {ist.frame}-{ist.ite}")
        ist.all_stalled.append((ist.frame, ist.ite))
        return True
    return False


def is_diverge(r,r_Axb):
    if (ist.ite < 5):
        return False

    if r[-1].dual/r[-5].dual>5:
        return True
    
    if r_Axb[-1]>r_Axb[0]:
        return True

    return False


def substep_xpbd(ist):
    gravity = ti.Vector(args.gravity)
    semi_euler(args.delta_t, ist.pos, ist.predict_pos, ist.old_pos, ist.vel, args.damping_coeff, gravity)
    reset_lagrangian(ist.lagrangian)
    r=[]
    for ist.ite in range(args.maxiter):
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
            args.omega
        )
        collsion_response(ist.pos)
        calc_dual_residual(ist.alpha_tilde, ist.lagrangian, ist.constraint, ist.dual_residual)
        dualr = np.linalg.norm(ist.residual.to_numpy())
        if ist.ite == 0:
            dualr0 = dualr.copy()
        toc = time.perf_counter()
        logging.info(f"{ist.frame}-{ist.ite} dual0:{dualr0:.2e} dual:{dualr:.2e} t:{toc-tic:.2e}s")
        r.append(ist.ResidualData(dualr, 0, toc-tic))
        if dualr < args.tol:
            logging.info("Converge: tol")
            break
        if dualr / dualr0 < args.rtol:
            logging.info("Converge: rtol")
            break
        # if is_stall(r):
        #     logging.warning("Stall detected, break")
        #     break
    ist.n_outer_all.append(ist.ite+1)
    update_vel(args.delta_t, ist.pos, ist.old_pos, ist.vel)




def ending(timer_loop, start_date, initial_frame):
    t_all = time.perf_counter() - timer_loop
    end_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.end_frame = ist.frame

    len_n_outer_all = len(ist.n_outer_all) if len(ist.n_outer_all) > 0 else 1
    sum_n_outer = sum(ist.n_outer_all)
    avg_n_outer = sum_n_outer / len_n_outer_all
    max_n_outer = max(ist.n_outer_all)
    max_n_outer_index = ist.n_outer_all.index(max_n_outer)

    n_outer_all_np = np.array(ist.n_outer_all, np.int32)    
    np.savetxt(args.out_dir+"/n_outer.txt", n_outer_all_np, fmt="%d")

    sim_time_with_export = time.perf_counter() - timer_loop
    sim_time = sim_time_with_export - ist.t_export_total
    avg_sim_time = sim_time / (args.end_frame - initial_frame)


    s = f"\n-------\n"+\
    f"Time: {(sim_time):.2f}s = {(sim_time)/60:.2f}min.\n" + \
    f"Time with exporting: {(sim_time_with_export):.2f}s = {sim_time_with_export/60:.2f}min.\n" + \
    f"Frame {initial_frame}-{args.end_frame}({args.end_frame-initial_frame} frames)."+\
    f"\nAvg: {avg_sim_time}s/frame."+\
    f"\nStart\t{start_date},\nEnd\t{end_date}."+\
    f"\nTime of exporting: {ist.t_export_total:.3f}s" + \
    f"\nSum n_outer: {sum_n_outer} \nAvg n_outer: {avg_n_outer:.1f}"+\
    f"\nMax n_outer: {max_n_outer} \nMax n_outer frame: {max_n_outer_index + initial_frame}." + \
    f"\nstalled at {ist.all_stalled}"+\
    f"\nmodel_path: {args.model_path}" + \
    f"\ndt={args.delta_t}" + \
    f"\nSolver: {args.solver_type}" + \
    f"\nout_dir: {args.out_dir}" 
    # logging.info(s)

    file_name = f"result/meta/{str(Path(args.out_dir).name)}.log"
    file_name2 = f"{args.out_dir}/meta.log"
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
    has_colored_L = [False]*ist.num_levels
    dir = str(Path(args.model_path).parent)
    for lv in range(ist.num_levels):
        path = dir+f'/coloring_L{lv}.txt'
        has_colored_L[lv] =  os.path.exists(path)
    has_colored = all(has_colored_L)
    if not has_colored:
        has_colored = True
    else:
        return

    from pyamg.graph import vertex_coloring
    tic = perf_counter()
    for i in range(ist.num_levels):
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
    process_dirs(args)

    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=args.out_dir + f'/{str(Path(args.out_dir).name)}.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # logger2 = logging.getLogger('logger2')
    # logger2.addHandler(logging.FileHandler(args.out_dir + f'/build_P_time.log', 'a'))

    start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.info(start_date)
    logging.info(args)

    if args.use_cuda:
        global extlib
        extlib = init_extlib(args,sim="soft")

    global ist
    ist = SoftBody(args.model_path)
    ist.initialize()
    
    if args.export_mesh:
        write_mesh(args.out_dir + f"/mesh/{ist.frame:04d}", ist.pos.to_numpy(), ist.model_tri)

    if args.solver_type != "XPBD":
        init_direct_fill_A(ist)

    if args.use_cuda:
        global amg_cuda
        amg_cuda = AmgCuda(args, ist, extlib, fill_A_csr_ti, fastFill_set, AMG_A, graph_coloring_v2)

    print(f"initialize time:", perf_counter()-tic)
    initial_frame = ist.frame
    ist.t_export_total = 0.0
    
    timer_all = perf_counter()
    step_pbar = tqdm.tqdm(total=args.end_frame, initial=initial_frame)
    try:
        while True:
            info("\n\n----------------------")
            info(f"frame {ist.frame}")
            t = perf_counter()
            ist.t_export = 0.0

            if args.solver_type == "XPBD":
                substep_xpbd(ist)
            else:
                substep_all_solver(ist)
            ist.frame += 1

            if args.export_mesh:
                tic = perf_counter()
                write_mesh(args.out_dir + f"/mesh/{ist.frame:04d}", ist.pos.to_numpy(), ist.model_tri)
                ist.t_export += perf_counter() - tic

            ist.t_export_total += ist.t_export

            info(f"step time: {perf_counter() - t:.2f} s")
            step_pbar.update(1)
                
            if ist.frame >= args.end_frame:
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
