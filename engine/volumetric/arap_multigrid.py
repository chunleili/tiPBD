"""
Modified YP multi-grid solver for ARAP
"""
import taichi as ti
from taichi.lang.ops import sqrt
import scipy.io as sio
import numpy as np
import logging
from logging import info
import scipy
from scipy.sparse import coo_matrix, spdiags, kron
from scipy.io import mmwrite
import sys, os, argparse

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_at", type=int, default=-1)
parser.add_argument("-s", "--save_at", type=int, default=-1)
parser.add_argument("-m", "--max_frame", type=int, default=-1)
parser.add_argument("-energy", "--log_energy_range", nargs=2, type=int, default=(-1, -1))
parser.add_argument("-residual", "--log_residual_range", nargs=2, type=int, default=(-1, -1))
parser.add_argument("-p", "--pause_at", type=int, default=-1)
parser.add_argument("-c", "--coarse_iterations", type=int, default=5)
parser.add_argument("-f", "--fine_iterations", type=int, default=5)
parser.add_argument("--model", type=str, default="cube")

ti.init(arch=ti.cpu)


class Meta:
    ...


meta = Meta()

# control parameters
meta.args = parser.parse_args()
meta.frame = 0
meta.use_multigrid = True
meta.max_frame = meta.args.max_frame
meta.log_energy_range = range(meta.args.log_energy_range[0], meta.args.log_energy_range[1])
meta.log_residual_range = range(meta.args.log_residual_range[0], meta.args.log_residual_range[1])
meta.frame_to_save = meta.args.save_at
meta.load_at = meta.args.load_at
meta.pause = False
meta.pause_at = meta.args.pause_at
meta.coarse_iterations, meta.fine_iterations = meta.args.coarse_iterations, meta.args.fine_iterations
if meta.coarse_iterations == 0 or meta.use_multigrid == False:
    meta.use_multigrid = False
    meta.coarse_iterations = 0

# physical parameters
meta.omega = 0.1  # SOR factor
meta.inv_mu = 1.0e-6
meta.h = 0.003
meta.inv_h2 = 1.0 / meta.h / meta.h
meta.gravity = ti.Vector([0.0, 0.0, 0.0])
meta.total_mass = 16000.0
# meta.mass_density = 2000.0
meta.damping_coeff = 1.0


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

        self.state = [
            self.pos,
            self.pos_mid,
            self.predict_pos,
            self.old_pos,
            self.vel,
            self.mass,
            self.inv_mass,
            self.tet_indices,
            self.display_indices,
            self.B,
            self.lagrangian,
            self.rest_volume,
            self.inv_V,
            self.alpha_tilde,
            self.par_2_tet,
            self.gradC,
            self.constraint,
            self.dpos,
            self.residual,
        ]


if meta.args.model == "bunny":
    meta.model_path = "data/model/bunny1k2k/"
    meta.fine_model_path = meta.model_path + "bunny2k"
    meta.coarse_model_path = meta.model_path + "bunny1k"
elif meta.args.model == "cube":
    meta.model_path = "data/model/cube/"
    meta.fine_model_path = meta.model_path + "fine"
    meta.coarse_model_path = meta.model_path + "coarse"

fine = ArapMultigrid(meta.fine_model_path)
coarse = ArapMultigrid(meta.coarse_model_path)


P = sio.mmread(meta.model_path + "P.mtx")
R = sio.mmread(meta.model_path + "R.mtx")


def update_fine_mesh():
    cpos_np = coarse.pos.to_numpy()
    fpos_np = P @ cpos_np
    fine.pos.from_numpy(fpos_np)


def update_coarse_mesh():
    fpos_np = fine.pos.to_numpy()
    cpos_np = R @ fpos_np
    coarse.pos.from_numpy(cpos_np)


@ti.kernel
def init_physics(
    pos_in: ti.types.ndarray(),
    tet_indices_in: ti.types.ndarray(),
    tri_indices_in: ti.types.ndarray(),
    pos_out: ti.template(),
    old_pos_out: ti.template(),
    vel_out: ti.template(),
    mass_out: ti.template(),
    tet_indices_out: ti.template(),
    B_out: ti.template(),
    inv_V_out: ti.template(),
    display_indices_out: ti.template(),
    par_2_tet: ti.template(),
    rest_volume: ti.template(),
    inv_mass: ti.template(),
):
    # init pos, old_pos, vel
    for i in pos_out:
        pos_out[i] = ti.Vector([pos_in[i, 0], pos_in[i, 1], pos_in[i, 2]])
        old_pos_out[i] = pos_out[i]
        vel_out[i] = ti.Vector([0, 0, 0])

    total_volume = 0.0
    for i in tet_indices_out:
        # init tet_indices
        a, b, c, d = (
            tet_indices_in[i, 0],
            tet_indices_in[i, 1],
            tet_indices_in[i, 2],
            tet_indices_in[i, 3],
        )
        tet_indices_out[i] = ti.Vector([a, b, c, d])
        # init B
        a, b, c, d = tet_indices_out[i]
        p0, p1, p2, p3 = pos_out[a], pos_out[b], pos_out[c], pos_out[d]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        B_out[i] = D_m.inverse()
        # init rest_volume and inv_V
        rest_volume[i] = 1.0 / 6.0 * ti.abs(D_m.determinant())
        inv_V_out[i] = 1.0 / rest_volume[i]
        total_volume += rest_volume[i]
        # init par_2_tet
        par_2_tet[a] = i
        par_2_tet[b] = i
        par_2_tet[c] = i
        par_2_tet[d] = i
    # init display_indices
    for i in range(tri_indices_in.shape[0]):
        display_indices_out[3 * i + 0] = tri_indices_in[i, 0]
        display_indices_out[3 * i + 1] = tri_indices_in[i, 1]
        display_indices_out[3 * i + 2] = tri_indices_in[i, 2]
    # init mass
    for i in tet_indices_out:
        a, b, c, d = (
            tet_indices_in[i, 0],
            tet_indices_in[i, 1],
            tet_indices_in[i, 2],
            tet_indices_in[i, 3],
        )
        mass_density = meta.total_mass / total_volume
        tet_mass = mass_density * rest_volume[i]
        avg_mass = tet_mass / 4.0
        mass_out[a] += avg_mass
        mass_out[b] += avg_mass
        mass_out[c] += avg_mass
        mass_out[d] += avg_mass
    for i in inv_mass:
        inv_mass[i] = 1.0 / mass_out[i]


@ti.kernel
def init_alpha_tilde(alpha_tilde: ti.template(), inv_V: ti.template()):
    for i in alpha_tilde:
        alpha_tilde[i] = meta.inv_h2 * meta.inv_mu * inv_V[i]


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
def project_constraints(
    mid_pos: ti.template(),
    tet_indices: ti.template(),
    inv_mass: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    pos: ti.template(),
    alpha_tilde: ti.template(),
    constraint: ti.template(),
    dpos: ti.template(),
):
    for i in pos:
        mid_pos[i] = pos[i]
        dpos[i] = ti.Vector([0.0, 0.0, 0.0])

    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = mid_pos[ia], mid_pos[ib], mid_pos[ic], mid_pos[id]
        D_s = ti.Matrix.cols([b - a, c - a, d - a])
        U, S, V = ti.svd(D_s @ B[i])
        if S[2, 2] < 0.0:  # S[2, 2] is the smallest singular value
            S[2, 2] *= -1.0
        constraint[i] = sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        if constraint[i] < 1e-12:
            continue
        g0, g1, g2, g3 = compute_gradient(U, S, V, B[i])
        l = (
            inv_mass[ia] * g0.norm_sqr()
            + inv_mass[ib] * g1.norm_sqr()
            + inv_mass[ic] * g2.norm_sqr()
            + inv_mass[id] * g3.norm_sqr()
        )
        dLambda = (constraint[i] - alpha_tilde[i] * lagrangian[i]) / (l + alpha_tilde[i])
        lagrangian[i] += dLambda
        pos[ia] -= meta.omega * inv_mass[ia] * dLambda * g0
        pos[ib] -= meta.omega * inv_mass[ib] * dLambda * g1
        pos[ic] -= meta.omega * inv_mass[ic] * dLambda * g2
        pos[id] -= meta.omega * inv_mass[id] * dLambda * g3


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


@ti.kernel
def init_random_position(pos: ti.template(), init_random_points: ti.types.ndarray()):
    for i in pos:
        pos[i] = ti.Vector(
            [
                init_random_points[i, 0],
                init_random_points[i, 1],
                init_random_points[i, 2],
            ]
        )


def log_energy(frame, filename_to_save):
    if frame in meta.log_energy_range:
        te, it, pe = compute_energy(fine.mass, fine.pos, fine.predict_pos, fine.tet_indices, fine.B, fine.alpha_tilde)
        info(f"energy:\t{te}")
        with open(filename_to_save, "a") as f:
            np.savetxt(f, np.array([te]), fmt="%.4e", delimiter="\t")


@ti.kernel
def compute_residual_kernel(
    constraint: ti.template(), alpha_tilde: ti.template(), lagrangian: ti.template(), residual: ti.template()
):
    for i in constraint:
        residual[i] = constraint[i] - alpha_tilde[i] * lagrangian[i]


def compute_residual() -> float:
    compute_residual_kernel(fine.constraint, fine.alpha_tilde, fine.lagrangian, fine.residual)
    r_norm = np.linalg.norm(fine.residual.to_numpy())
    return r_norm


def log_residual(frame, filename_to_save):
    if frame in meta.log_residual_range:
        r_norm = compute_residual()
        logging.info("residual:\t{}".format(r_norm))
        with open(filename_to_save, "a") as f:
            np.savetxt(f, np.array([r_norm]), fmt="%.4e", delimiter="\t")


def save_state(filename):
    state = fine.state + coarse.state
    for i in range(1, len(state)):
        state[i] = state[i].to_numpy()
    np.savez(filename, *state)
    logging.info(f"saved state to '{filename}', totally saved {len(state)} variables")


def load_state(filename):
    npzfile = np.load(filename)
    state = fine.state + coarse.state
    meta.frame = int(npzfile["arr_0"])
    for i in range(1, len(state)):
        state[i].from_numpy(npzfile["arr_" + str(i)])

    logging.info(f"loaded state from '{filename}', totally loaded {len(state)} variables")


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format=" %(levelname)s %(message)s")

    init_physics(
        fine.model_pos,
        fine.model_tet,
        fine.model_tri,
        fine.pos,
        fine.old_pos,
        fine.vel,
        fine.mass,
        fine.tet_indices,
        fine.B,
        fine.inv_V,
        fine.display_indices,
        fine.par_2_tet,
        fine.rest_volume,
        fine.inv_mass,
    )
    init_physics(
        coarse.model_pos,
        coarse.model_tet,
        coarse.model_tri,
        coarse.pos,
        coarse.old_pos,
        coarse.vel,
        coarse.mass,
        coarse.tet_indices,
        coarse.B,
        coarse.inv_V,
        coarse.display_indices,
        coarse.par_2_tet,
        coarse.rest_volume,
        coarse.inv_mass,
    )

    init_style = "enlarge"

    if init_style == "random":
        # random init
        random_val = np.random.rand(fine.pos.shape[0], 3)
        fine.pos.from_numpy(random_val)
        coarse.pos.from_numpy(random_val)
    elif init_style == "enlarge":
        # init by enlarge 1.5x
        fine.pos.from_numpy(fine.model_pos * 1.5)
        coarse.pos.from_numpy(coarse.model_pos * 1.5)

    init_alpha_tilde(fine.alpha_tilde, fine.inv_V)
    init_alpha_tilde(coarse.alpha_tilde, coarse.inv_V)
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
    energy_filename = "result/log/totalEnergy_" + (suffix) + ".txt"
    residual_filename = "result/log/residual_" + (suffix) + ".txt"
    if os.path.exists(energy_filename):
        os.remove(energy_filename)
    if os.path.exists(residual_filename):
        os.remove(residual_filename)

    save_state_filename = "result/save/frame_"
    if meta.load_at != -1:
        meta.filename_to_load = save_state_filename + str(meta.load_at) + ".npz"
        load_state(meta.filename_to_load)

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
        meta.coarse_iterations = gui.slider_int("coarse_iterations", meta.coarse_iterations, 0, 50)
        meta.fine_iterations = gui.slider_int("fine_iterations", meta.fine_iterations, 0, 50)
        gui.text("total iterations: {}".format(meta.coarse_iterations + meta.fine_iterations))

        if meta.frame == meta.frame_to_save:
            save_state(save_state_filename + str(meta.frame))

        if not meta.pause:
            info(f"######## frame {meta.frame} ########")
            semi_euler(meta.h, fine.pos, fine.predict_pos, fine.old_pos, fine.vel, meta.damping_coeff)
            if meta.use_multigrid:
                update_coarse_mesh()
            reset_lagrangian(coarse.lagrangian)
            for ite in range(meta.coarse_iterations):
                log_energy(meta.frame, energy_filename)
                project_constraints(
                    coarse.pos_mid,
                    coarse.tet_indices,
                    coarse.inv_mass,
                    coarse.lagrangian,
                    coarse.B,
                    coarse.pos,
                    coarse.alpha_tilde,
                    coarse.constraint,
                    coarse.dpos,
                )
                log_residual(meta.frame, residual_filename)
                collsion_response(coarse.pos)
                update_fine_mesh()
            reset_lagrangian(fine.lagrangian)
            for ite in range(meta.fine_iterations):
                if ite == 0:
                    log_residual(meta.frame, residual_filename)
                log_energy(meta.frame, energy_filename)
                project_constraints(
                    fine.pos_mid,
                    fine.tet_indices,
                    fine.inv_mass,
                    fine.lagrangian,
                    fine.B,
                    fine.pos,
                    fine.alpha_tilde,
                    fine.constraint,
                    fine.dpos,
                )
                log_residual(meta.frame, residual_filename)
                collsion_response(fine.pos)

                update_velocity(meta.h, fine.pos, fine.old_pos, fine.vel)

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
