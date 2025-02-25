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
from time import perf_counter
from pathlib import Path

sys.path.append(os.getcwd())
from compute_R_acc import compute_mapping, compute_mapping_v2


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_at", type=int, default=-1)
parser.add_argument("-s", "--save_at", type=int, default=-1)
parser.add_argument("-m", "--max_frame", type=int, default=-1)
parser.add_argument("-e", "--log_energy",  type=int, default=0)
parser.add_argument("-r", "--log_residual", type=int, default=0)
parser.add_argument("-p", "--pause_at", type=int, default=-1)
parser.add_argument("-c", "--coarse_iterations", type=int, default=5)
parser.add_argument("-f", "--fine_iterations", type=int, default=2)
parser.add_argument("-it", "--mg_maxiter", type=int, default=1)
parser.add_argument("--model", type=str, default="bunny", choices=["bunny", "cube","beam"])
parser.add_argument("--fine_model_path", type=str, default="")
parser.add_argument("--coarse_model_path", type=str, default="")
parser.add_argument("--omega", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=1e20)
parser.add_argument("--dt", type=float, default=33e-3)
parser.add_argument("--damping_coeff", type=float, default=1.0)
parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0, 0.0))
parser.add_argument("--total_mass", type=float, default=16000.0)
parser.add_argument("--use_multigrid", type=int, default=False)
parser.add_argument("--init_style", type=str, default="", choices=["","random", "enlarge","squash","zero","freefall","rest","fixleft"])
parser.add_argument("--silence", type=int, default=1)
parser.add_argument("--out_dir", type=str, default="result/latest")
parser.add_argument("--export_mesh", type=int, default=False)


ti.init(arch=ti.gpu)


class Meta:
    ...


meta = Meta()

# control parameters
meta.args = parser.parse_args()
meta.frame = 0
meta.use_multigrid = meta.args.use_multigrid
meta.max_frame = meta.args.max_frame
# meta.log_energy_range = range(*meta.args.log_energy_range)
# meta.log_residual_range = range(*meta.args.log_residual_range)
meta.frame_to_save = meta.args.save_at
meta.load_at = meta.args.load_at
meta.pause = True
meta.pause_at = meta.args.pause_at
meta.coarse_iterations, meta.fine_iterations = meta.args.coarse_iterations, meta.args.fine_iterations
# if meta.coarse_iterations == 0 or meta.use_multigrid == False:
#     meta.use_multigrid = False
#     meta.coarse_iterations = 0

# physical parameters
meta.omega = meta.args.omega  # SOR factor, default 0.1
meta.mu = meta.args.mu  # Lame's second parameter, default 1e6
meta.h = meta.args.dt  # time step size, default 3e-3
meta.inv_h2 = 1.0 / meta.h / meta.h
meta.gravity = ti.Vector(meta.args.gravity)  # gravity, default (0, 0, 0)
meta.damping_coeff = meta.args.damping_coeff  # damping coefficient, default 1.0
meta.total_mass = meta.args.total_mass  # total mass, default 16000.0
# meta.mass_density = 2000.0


def timeit(method):
    def timed(*args, **kw):
        ts = perf_counter()
        result = method(*args, **kw)
        te = perf_counter()
        logging.info(f"    {method.__name__} took: {(te-ts)*1000:.1f}ms")
        return result
    return timed


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


class ArapHpbd:
    def __init__(self, path):
        self.model_pos, self.model_tet, self.model_tri = read_tetgen(path)
        self.NV = len(self.model_pos)
        self.NT = len(self.model_tet)
        self.NF = len(self.model_tri)

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
        self.alpha_tilde = ti.field(float, self.NT)

        self.par_2_tet = ti.field(int, self.NV)
        self.constraint = ti.field(ti.f32, shape=(self.NT))
        self.residual = ti.field(ti.f32, shape=self.NT)

        self.state = [
            self.pos,
            self.vel,
        ]

def load_model():
    if meta.args.fine_model_path != "" and meta.args.coarse_model_path != "":
        meta.fine_model_path = meta.args.fine_model_path
        meta.coarse_model_path = meta.args.coarse_model_path
        meta.model_path = str(Path(meta.fine_model_path).parent)
    elif meta.args.model == "bunny":
        meta.model_path = "data/model/bunny85w/"
        meta.fine_model_path = meta.model_path + "bunny85w"
        meta.coarse_model_path = meta.model_path + "bunny5k"
    elif meta.args.model == "cube":
        meta.model_path = "data/model/cube_64k/"
        meta.fine_model_path = meta.model_path + "fine"
        meta.coarse_model_path = meta.model_path + "coarse"
    elif meta.args.model == "beam":
        meta.model_path = "data/model/beam458k/"
        meta.fine_model_path = meta.model_path + "beam458k"
        meta.coarse_model_path = meta.model_path + "beam0.6k"

load_model()

fine = ArapHpbd(meta.fine_model_path)
coarse = ArapHpbd(meta.coarse_model_path)


print(">> Start to compute coarse and fine mapping...")
(
    coarse_in_fine_tet_indx,
    coarse_in_fine_tet_coord,
    fine_in_coarse_tet_indx,
    fine_in_coarse_tet_coord,
) = compute_mapping(coarse.model_pos, coarse.model_tet, fine.model_pos, fine.model_tet)

# extra variable for prolongation and restriction
cage_idx = ti.field(int, fine.NV) # cage index(coarse tet index) for each vertex in fine mesh
uvw = ti.Vector.field(3, float, fine.NV) # barycentric coordinate  for each fine vertex 
cage_idx.from_numpy(fine_in_coarse_tet_indx)
uvw.from_numpy(fine_in_coarse_tet_coord)


cage_idx_c2f = ti.field(int, coarse.NV) #coarse_in_fine_tet_indx
uvw_c2f = ti.Vector.field(3, float, coarse.NV) #coarse_in_fine_tet_coord
cage_idx_c2f.from_numpy(coarse_in_fine_tet_indx)
uvw_c2f.from_numpy(coarse_in_fine_tet_coord)


# print(">> Start to compute coarse and fine mapping...")
# (
#     coarse2fine_nearest_vert,
#     fine_in_coarse_tet_indx,
#     fine_in_coarse_tet_coord,
# ) = compute_mapping_v2(coarse.model_pos, coarse.model_tet, fine.model_pos)

# c2f_nearest = ti.field(int, coarse.NV) # nearest vertex in fine mesh for each vertex in coarse mesh
# c2f_nearest.from_numpy(coarse2fine_nearest_vert) #this way momentum will be not conserved, causing rotation

# P = sio.mmread(meta.model_path + "P.mtx")
# R = sio.mmread(meta.model_path + "R.mtx")


# @timeit
def update_fine_mesh():
    # cpos_np = coarse.pos.to_numpy()
    # fpos_np = P @ cpos_np
    # fine.pos.from_numpy(fpos_np)
    update_fine_mesh_mfree_kernel()


# xf = P@ xc in matrix free version
@ti.kernel
def update_fine_mesh_mfree_kernel():
    for i in range(fine.NV):
        # v1 direct coarset cage
        c1,c2,c3,c4 = coarse.tet_indices[cage_idx[i]]
        u,v,w = uvw[i]
        fine.pos[i] = (1-u-v-w)*coarse.pos[c1] + u*coarse.pos[c2] + v*coarse.pos[c3] + w*coarse.pos[c4]

        # # v2 dp coarse cage: Not converge
        # c1,c2,c3,c4 = coarse.tet_indices[cage_idx[i]]
        # u,v,w = uvw[i]
        # dp1 = coarse.pos[c1] - coarse.old_pos[c1]
        # dp2 = coarse.pos[c2] - coarse.old_pos[c2]
        # dp3 = coarse.pos[c3] - coarse.old_pos[c3]
        # dp4 = coarse.pos[c4] - coarse.old_pos[c4]
        # fine.pos[i] += (1-u-v-w)*dp1 + u*dp2 + v*dp3 + w*dp4

    
# @timeit
def update_coarse_mesh():
    # fpos_np = fine.pos.to_numpy()
    # cpos_np = R @ fpos_np
    # coarse.pos.from_numpy(cpos_np)
    update_coarse_mesh_mfree_kernel()

 
# xc = R@ xf in matrix free version, 
@ti.kernel
def update_coarse_mesh_mfree_kernel():
    for i in range(coarse.NV):
        # v1 direct fine cage
        # c1,c2,c3,c4 = fine.tet_indices[cage_idx_c2f[i]]
        # u,v,w = uvw_c2f[i]
        # coarse.pos[i] = (1-u-v-w)*fine.pos[c1] + u*fine.pos[c2] + v*fine.pos[c3] + w*fine.pos[c4]

        # v2 dp fine cage
        c1,c2,c3,c4 = fine.tet_indices[cage_idx_c2f[i]]
        u,v,w = uvw_c2f[i]
        dp1 = fine.pos[c1] - fine.old_pos[c1]
        dp2 = fine.pos[c2] - fine.old_pos[c2]
        dp3 = fine.pos[c3] - fine.old_pos[c3]
        dp4 = fine.pos[c4] - fine.old_pos[c4]
        coarse.pos[i] += (1-u-v-w)*dp1 + u*dp2 + v*dp3 + w*dp4

        # v3 direct nearest
        # coarse.pos[i] = fine.pos[c2f_nearest[i]] #direct nearest will be worse in mementum imbalance

        # v4 dp nearest
        # dp = fine.pos[c2f_nearest[i]] - fine.old_pos[c2f_nearest[i]]
        # coarse.pos[i] += dp
    

def get_bbox(pos):
    lowest_x = np.min(pos[:, 0])
    highest_x = np.max(pos[:, 0])
    lowest_y = np.min(pos[:, 1])
    highest_y = np.max(pos[:, 1])
    lowest_z = np.min(pos[:, 2])
    highest_z = np.max(pos[:, 2])
    bbox = np.array(
        [
            [lowest_x, lowest_y, lowest_z],
            [highest_x, highest_y, highest_z],
        ]
    )
    return bbox


def rescale(pos):
    print("rescaling the model to unit cube")
    bbox = get_bbox(pos)
    center = (bbox[0] + bbox[1]) / 2
    scale = 1.0 / np.max(bbox[1] - bbox[0])
    print("center", center)
    print("scale", scale)
    pos -= center
    pos *= scale
    return pos
    

def init_model(instance):
    instance.pos.from_numpy(instance.model_pos)
    instance.tet_indices.from_numpy(instance.model_tet)
    instance.display_indices.from_numpy(instance.model_tri.flatten())
    instance.bbox = get_bbox(instance.model_pos)
    print("\nbbox\n", instance.bbox)
    instance.lowest_y = instance.bbox[0, 1]
    print("lowest_y", instance.lowest_y)
    instance.model_pos = rescale(instance.model_pos)
    instance.pos.from_numpy(instance.model_pos)



@ti.kernel
def init_physics(
    pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    rest_volume: ti.template(),
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
        total_volume += rest_volume[i]

    mass_density = meta.total_mass / total_volume
    # mass_density = 1
    print("mass_density", mass_density)
    # init mass
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
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
        alpha_tilde[i] = meta.inv_h2 / meta.mu / rest_volume[i]

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
    inv_mass: ti.template(),
):
    for i in pos:
        if inv_mass[i] != 0.0:
            vel[i] += h * meta.gravity
            vel[i] *= damping_coeff
            old_pos[i] = pos[i]
            pos[i] += h * vel[i]
            predict_pos[i] = pos[i]


@ti.kernel
def update_velocity(h: ti.f32, pos: ti.template(), old_pos: ti.template(), vel: ti.template(), inv_mass: ti.template()):
    for i in pos:
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h


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
        denorminator = (
            inv_mass[p0] * g0.norm_sqr()
            + inv_mass[p1] * g1.norm_sqr()
            + inv_mass[p2] * g2.norm_sqr()
            + inv_mass[p3] * g3.norm_sqr()
        )
        dlambda = -(constraint[t] + alpha_tilde[t] * lagrangian[t]) / (denorminator + alpha_tilde[t])

        lagrangian[t] += dlambda

        pos[p0] += meta.omega * inv_mass[p0] * dlambda * g0
        pos[p1] += meta.omega * inv_mass[p1] * dlambda * g1
        pos[p2] += meta.omega * inv_mass[p2] * dlambda * g2
        pos[p3] += meta.omega * inv_mass[p3] * dlambda * g3

        residual[t] = constraint[t] + alpha_tilde[t] * lagrangian[t]


@ti.kernel
def collsion_response(pos: ti.template(), old_pos:ti.template(), ground_pos: ti.f32, inv_mass: ti.template()):
    for i in pos:
        if inv_mass[i] != 0.0:
            if pos[i][1] < meta.lowest_y-0.1:
                pos[i] = old_pos[i]
                pos[i][1] = meta.lowest_y-0.1


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


def log_energy(frame, filename_to_save=""):
    if meta.args.log_energy:
        te, it, pe = compute_energy(fine.mass, fine.pos, fine.predict_pos, fine.tet_indices, fine.B, fine.alpha_tilde)
        info(f"energy:\t{te}")
        if filename_to_save != "":
            with open(filename_to_save, "a") as f:
                f.write(f"{frame}\t{te:.2e}\n")
        return te

@timeit
def log_residual(frame, filename_to_save):
    if meta.args.log_residual:
        r_norm = np.linalg.norm(fine.residual.to_numpy())
        logging.info("residual:\t{}".format(r_norm))
        with open(filename_to_save, "a") as f:
            f.write(f"{frame}\t{r_norm:.2e}\n")
        return r_norm


def save_state(filename):
    state = fine.state + coarse.state
    for i in range(0, len(state)):
        state[i] = state[i].to_numpy()
    np.savez(filename, *state)
    logging.info(f"saved state to '{filename}', totally saved {len(state)} variables")


def load_state(filename):
    npzfile = np.load(filename)
    state = fine.state + coarse.state
    for i in range(0, len(state)):
        state[i].from_numpy(npzfile["arr_" + str(i)])
    fine.lagrangian.fill(0.0)
    coarse.lagrangian.fill(0.0)
    logging.info(f"loaded state from '{filename}', totally loaded {len(state)} variables")



def fixleft(ist):
    p = ist.model_pos
    # fix the beam at the left end
    # find the left end postion
    xmin = np.min(p, axis=0)[0]
    xmax = np.max(p, axis=0)[0]
    xsize = xmax - xmin 
    # a small region within the left end
    endregion = (xmin - xsize*0.01, xmin + xsize*0.01)
    # find the end particles where x is within the region
    ist.fixed_particles = np.where((p[:, 0] > endregion[0]) & (p[:, 0] < endregion[1]))[0]
    # set those particles inv_mass to 0
    ist.inv_mass_np = ist.inv_mass.to_numpy()
    # ist.inv_mass_np = args.pmass * np.ones(ist.NV, dtype=np.float32)
    ist.inv_mass_np[ist.fixed_particles] = 0.0
    ist.inv_mass.from_numpy(ist.inv_mass_np)
    meta.gravity = ti.Vector([0, -9.8, 0])


def reinit(init_style=""):
    meta.frame=0
    if init_style == "random":
        random_val = np.random.rand(fine.pos.shape[0], 3)
        fine.pos.from_numpy(random_val)
        coarse.pos.from_numpy(random_val)
    elif init_style == "enlarge":
        # init by enlarge 1.5x
        fine.pos.from_numpy(fine.model_pos * 1.5)
        coarse.pos.from_numpy(coarse.model_pos * 1.5)
    elif init_style == "squash":
        p = fine.model_pos.copy()
        p[:, 1] *= 0
        fine.pos.from_numpy(p)
        p = coarse.model_pos.copy()
        p[:, 1] *= 0
        coarse.pos.from_numpy(p)
    elif init_style == "zero":
        fine.pos.from_numpy(fine.model_pos * 0)
        coarse.pos.from_numpy(coarse.model_pos * 0)
    elif init_style == "rest":
        fine.pos.from_numpy(fine.model_pos)
        coarse.pos.from_numpy(coarse.model_pos)
    elif init_style == "fixleft":
        fixleft(fine)
        fixleft(coarse)
    # update_coarse_mesh()
    print(f"reinit {init_style}")


def write_mesh(filename, pos, tri):
    import meshio
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )
    mesh.write(filename, binary=True)
    return mesh


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if(meta.args.silence):
        logging.getLogger().setLevel(logging.ERROR)
    meta.out_dir = Path(meta.args.out_dir)
    Path(meta.out_dir).mkdir(parents=True, exist_ok=True)
    Path(meta.out_dir/"state").mkdir(parents=True, exist_ok=True)
    Path(meta.out_dir/"A").mkdir(parents=True, exist_ok=True)
    Path(meta.out_dir/"mesh").mkdir(parents=True, exist_ok=True)
    Path(meta.out_dir/"r").mkdir(parents=True, exist_ok=True)

    init_model(fine)
    init_model(coarse)

    init_physics(
        fine.pos,
        fine.old_pos,
        fine.vel,
        fine.tet_indices,
        fine.B,
        fine.rest_volume,
        fine.mass,
        fine.inv_mass,
        fine.alpha_tilde,
        fine.par_2_tet,
    )
    init_physics(
        coarse.pos,
        coarse.old_pos,
        coarse.vel,
        coarse.tet_indices,
        coarse.B,
        coarse.rest_volume,
        coarse.mass,
        coarse.inv_mass,
        coarse.alpha_tilde,
        coarse.par_2_tet,
    )

    print("saving rest state and 0 state(deformed)")
    save_state("result/latest/state/rest.npz") #rest state
    reinit(meta.args.init_style)#initial deform 
    save_state("result/latest/state/0.npz") # initial state

    window = ti.ui.Window("3D ARAP FEM XPBD", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.lookat(0.5,0.5,1)
    camera.position(0.5, 0.5, 4)
    camera.fov(45)
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
    gui = window.get_gui()
    wire_frame = True
    should_reset = False

    if meta.use_multigrid:
        suffix = "mg"
        info("#############################################")
        info("########## Using Multi-Grid Solver ##########")
        info("#############################################")
    else:
        suffix = "onlyfine"
        info("#############################################")
        info("########## Using Only Fine Solver ###########")
        info("#############################################")
    energy_filename = f"{meta.out_dir}/r/energy_{suffix}" + ".txt"
    residual_filename = f"{meta.out_dir}/r/residual_{suffix}" + ".txt"
    Path(energy_filename).write_text(f"")
    Path(energy_filename).write_text(f"")


    save_state_filename = f"{meta.out_dir}/state/"
    if meta.load_at != -1:
        meta.filename_to_load = save_state_filename + str(meta.load_at) + ".npz"
        load_state(meta.filename_to_load)
    
    timer_frame = []
    timer_coarse = []
    timer_fine = []
    timer_restrict = []
    timer_prolong = []
    while window.running:
        scene.ambient_light((0.8, 0.8, 0.8))
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        if window.is_pressed(ti.ui.ESCAPE):
            window.running = False

        meta.pause = gui.checkbox("pause", meta.pause)
        if meta.frame == meta.pause_at:
            meta.pause = True

        gui.text("frame {}".format(meta.frame))
        wire_frame = gui.checkbox("wireframe", wire_frame)
        meta.args.export_mesh = gui.checkbox("export mesh", meta.args.export_mesh)
        meta.args.log_residual = gui.checkbox("log residual", meta.args.log_residual)
        meta.args.log_energy = gui.checkbox("log energy", meta.args.log_energy)
        should_reset = gui.button("reset")
        squash = gui.button("squash")
        zero = gui.button("zero")
        random = gui.button("random")
        meta.use_multigrid = gui.checkbox("multigrid", meta.use_multigrid)
        meta.coarse_iterations = gui.slider_int("coarse_iterations", meta.coarse_iterations, 0, 50)
        meta.fine_iterations = gui.slider_int("fine_iterations", meta.fine_iterations, 0, 50)
        meta.args.mg_maxiter = gui.slider_int("mg_maxiter", meta.args.mg_maxiter, 0, 50)
        gui.text(f"F #tets: {fine.NT} #verts: {fine.NV}")
        gui.text(f"C #tets: {coarse.NT} #verts: {coarse.NV}")
        gui.text(f"dt={meta.h*1000:.1f}ms mu={meta.mu:.2e} omega={meta.omega:.2f} ")
        gui.text(f"camera: {camera.curr_lookat} {camera.curr_position}")

        if meta.frame == meta.frame_to_save:
            save_state(save_state_filename + str(meta.frame))
        
        if should_reset:
            load_state(f"{meta.out_dir}/state/rest.npz")
            reinit(meta.args.init_style)
            should_reset = False
        if squash:
            reinit("squash")
            squash = False
        if zero:
            reinit("zero")
            zero = False
        if random:
            reinit("random")
            random = False

        if not meta.pause:
            s = f"frame {meta.frame} "
            tic_frame = perf_counter()
            semi_euler(meta.h, fine.pos, fine.predict_pos, fine.old_pos, fine.vel, meta.damping_coeff, fine.inv_mass)
            for meta.mgIter in range(meta.args.mg_maxiter):
                if meta.mgIter == 0:
                    if meta.args.log_residual:
                        log_residual(meta.frame, residual_filename)
                    if meta.args.log_energy:
                        log_energy(meta.frame, energy_filename)
                if meta.use_multigrid:
                    tic_restrict = perf_counter()
                    update_coarse_mesh() # Restriction
                    toc_restrict = perf_counter()
                    timer_restrict.append(toc_restrict - tic_restrict)
                    reset_lagrangian(coarse.lagrangian) # coarse xpbd(coarse solve)
                    tic_coarse = perf_counter()
                    for ite in range(meta.coarse_iterations):
                        project_constraints(
                            coarse.pos_mid,
                            coarse.tet_indices,
                            coarse.inv_mass,
                            coarse.lagrangian,
                            coarse.B,
                            coarse.pos,
                            coarse.alpha_tilde,
                            coarse.constraint,
                            coarse.residual,
                        )
                    toc_coarse = perf_counter()
                    timer_coarse.append(toc_coarse - tic_coarse)
                    tic_prolong = perf_counter()
                    update_fine_mesh() # Prolongation
                    toc_prolong = perf_counter()
                    timer_prolong.append(toc_prolong - tic_prolong)
                    s+= f"coarse: {(timer_coarse[-1])*1000:.1f}ms "
                tic_fine = perf_counter()
                reset_lagrangian(fine.lagrangian) # fine xpbd(postsmoother)
                for ite in range(meta.fine_iterations):
                    project_constraints(
                        fine.pos_mid,
                        fine.tet_indices,
                        fine.inv_mass,
                        fine.lagrangian,
                        fine.B,
                        fine.pos,
                        fine.alpha_tilde,
                        fine.constraint,
                        fine.residual,
                    )

                if meta.args.log_residual:
                    dualr=log_residual(meta.frame, residual_filename)
                    gui.text(f"residual: {dualr:.1e}")
                if meta.args.log_energy:
                    energy = log_energy(meta.frame, energy_filename)
                    gui.text(f"energy: {energy:.1e}")

            # collsion_response(fine.pos, fine.old_pos, 0.0, fine.inv_mass)
            update_velocity(meta.h, fine.pos, fine.old_pos, fine.vel, fine.inv_mass)
            toc_fine = perf_counter()
            timer_fine.append(toc_fine - tic_fine)
            s+= f"fine: {(timer_fine[-1])*1000:.1f}ms "
            toc_frame = perf_counter()
            timer_frame.append(toc_frame - tic_frame)
            s+=f"t: {(timer_frame[-1])*1000:.1f}ms"
            if not meta.args.silence:
                logging.info(s)
            meta.frame += 1

        if timer_frame:
            gui.text(f"{timer_frame[-1] * 1000:.1f} ms/frame")
            if meta.use_multigrid:
                if timer_coarse :
                    gui.text(f"C:{timer_coarse[-1] * 1000:.1f} ms")
                if timer_restrict:
                    gui.text(f"R:{timer_restrict[-1] * 1000:.1f} ms")
                if timer_prolong:
                    gui.text(f"P:{timer_prolong[-1] * 1000:.1f} ms")
            gui.text(f"F:{timer_fine[-1] * 1000:.1f} ms")
            gui.text(f"FPS(physics): {1.0/timer_frame[-1]:.1f}")

        if meta.frame == meta.max_frame:
            window.running = False
            break

        if meta.args.export_mesh and not meta.pause:
            logging.info(f"exporting {meta.frame:04d}.ply")
            write_mesh(meta.out_dir / f"mesh/{meta.frame:04d}.ply", fine.pos.to_numpy(), fine.model_tri)

        scene.mesh(fine.pos, fine.display_indices, color=(1.0, 0.5, 0.5), show_wireframe=wire_frame)

        if meta.use_multigrid:
            scene.mesh(coarse.pos, coarse.display_indices, color=(0.0, 0.5, 1.0), show_wireframe=wire_frame)

        canvas.scene(scene)
        window.show()
    timer_frame = np.array(timer_frame)
    logging.info(f"average frame time: {np.mean(timer_frame)*1000:.1f}ms")


if __name__ == "__main__":
    main()
