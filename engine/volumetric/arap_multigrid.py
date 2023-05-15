"""
Modified YP multi-grid solver for ARAP
"""
import taichi as ti
from taichi.lang.ops import sqrt
import scipy.io as sio
import numpy as np
import logging
from logging import info

import sys,os
sys.path.append(os.getcwd())
from engine.mesh_io import read_tetgen
# from engine.log import log_energy

ti.init(arch=ti.cpu)

omega = 0.1  # SOR factor
inv_mu = 1.0e-6
h = 0.003
inv_h2 = 1.0 / h / h
gravity = ti.Vector([0.0, 0.0, 0.0])
DIM = 3
coarse_iterations, fine_iterations = 5, 5
only_fine_iterations = coarse_iterations + fine_iterations

mass_density = 2000
damping_coeff = 0.99

fine_model_pos, fine_model_inx, fine_model_tri = read_tetgen("data/model/cube_64k/fine")
fNV = len(fine_model_pos)
fNT = len(fine_model_inx)
fNF = len(fine_model_tri)
fpos = ti.Vector.field(DIM, float, fNV)
fpos_mid = ti.Vector.field(DIM, float, fNV)
fpredict_pos = ti.Vector.field(DIM, float, fNV)
fold_pos = ti.Vector.field(DIM, float, fNV)
fvel = ti.Vector.field(DIM, float, fNV)  # velocity of particles
fmass = ti.field(float, fNV)  # mass of particles
ftet_indices = ti.Vector.field(4, int, fNT)
fdisplay_indices = ti.field(ti.i32, fNF * 3)
fB = ti.Matrix.field(DIM, DIM, float, fNT)  # D_m^{-1}
flagrangian = ti.field(float, fNT)  # lagrangian multipliers
finv_V = ti.field(float, fNT)  # volume of each tet
falpha_tilde = ti.field(float, fNT)

coarse_model_pos, coarse_model_inx, coarse_model_tri = read_tetgen(
    "data/model/cube_64k/coarse")
cNV = len(coarse_model_pos)
cNT = len(coarse_model_inx)
cNF = len(coarse_model_tri)
cpos = ti.Vector.field(DIM, float, cNV)
cpos_mid = ti.Vector.field(DIM, float, cNV)
cpredict_pos = ti.Vector.field(DIM, float, cNV)
cold_pos = ti.Vector.field(DIM, float, cNV)
cvel = ti.Vector.field(DIM, float, cNV)  # velocity of particles
cmass = ti.field(float, cNV)  #inverse mass of particles
ctet_indices = ti.Vector.field(4, int, cNT)
cdisplay_indices = ti.field(ti.i32, cNF * 3)
cB = ti.Matrix.field(DIM, DIM, float, cNT)  # D_m^{-1}
clagrangian = ti.field(float, cNT)  # lagrangian multipliers
cinv_V = ti.field(float, cNT)  # volume of each tet
calpha_tilde = ti.field(float, cNT)

P = sio.mmread("data/model/cube_64k/P.mtx")


def update_fine_mesh():
    cpos_np = cpos.to_numpy()
    fpos_np = P @ cpos_np
    fpos.from_numpy(fpos_np)


R = sio.mmread("data/model/cube_64k/R.mtx")


def update_coarse_mesh():
    fpos_np = fpos.to_numpy()
    cpos_np = R @ fpos_np
    cpos.from_numpy(cpos_np)


@ti.kernel
def init_pos(pos_in: ti.types.ndarray(), tet_indices_in: ti.types.ndarray(),
             tri_indices_in: ti.types.ndarray(), pos_out: ti.template(),
             old_pos_out: ti.template(), vel_out: ti.template(),
             mass_out: ti.template(), tet_indices_out: ti.template(),
             B_out: ti.template(), inv_V_out: ti.template(),
             display_indices_out: ti.template(), NF: int):
    for i in pos_out:
        pos_out[i] = ti.Vector([pos_in[i, 0], pos_in[i, 1], pos_in[i, 2]])
        old_pos_out[i] = pos_out[i]
        vel_out[i] = ti.Vector([0, 0, 0])
    for i in tet_indices_out:
        a, b, c, d = tet_indices_in[i, 0], tet_indices_in[
            i, 1], tet_indices_in[i, 2], tet_indices_in[i, 3]
        tet_indices_out[i] = ti.Vector([a, b, c, d])
        a, b, c, d = tet_indices_out[i]
        p0, p1, p2, p3 = pos_out[a], pos_out[b], pos_out[c], pos_out[d]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        rest_volume = 1.0 / 6.0 * ti.abs(D_m.determinant())
        mass = mass_density * rest_volume
        avg_mass = mass / 4.0
        mass_out[a] += avg_mass
        mass_out[b] += avg_mass
        mass_out[c] += avg_mass
        mass_out[d] += avg_mass
        inv_V_out[i] = 1.0 / rest_volume
        B_out[i] = D_m.inverse()
    for i in range(NF):
        display_indices_out[3 * i + 0] = tri_indices_in[i, 0]
        display_indices_out[3 * i + 1] = tri_indices_in[i, 1]
        display_indices_out[3 * i + 2] = tri_indices_in[i, 2]


@ti.kernel
def init_alpha_tilde(alpha_tilde: ti.template(), inv_V: ti.template()):
    for i in alpha_tilde:
        alpha_tilde[i] = inv_h2 * inv_mu * inv_V[i]


@ti.kernel
def resetLagrangian(lagrangian: ti.template()):
    for i in lagrangian:
        lagrangian[i] = 0.0


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.func
def computeGradient(U, S, V, B):
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] - 1)**2)

    # (dcdS00, dcdS11, dcdS22)
    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1, S[2, 2] - 1])
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
    dcdF = ti.Vector([
        dsdF00.dot(dcdS),
        dsdF10.dot(dcdS),
        dsdF20.dot(dcdS),
        dsdF01.dot(dcdS),
        dsdF11.dot(dcdS),
        dsdF21.dot(dcdS),
        dsdF02.dot(dcdS),
        dsdF12.dot(dcdS),
        dsdF22.dot(dcdS)
    ])
    g1 = dFdp1T @ dcdF
    g2 = dFdp2T @ dcdF
    g3 = dFdp3T @ dcdF
    g0 = -g1 - g2 - g3
    return g0, g1, g2, g3


@ti.kernel
def semiEuler(h: ti.f32, pos: ti.template(), predic_pos: ti.template(),
              old_pos: ti.template(), vel: ti.template(), damping_coeff: ti.f32):
    # semi-Euler update pos & vel
    for i in pos:
        vel[i] += h * gravity
        vel[i] *= damping_coeff
        old_pos[i] = pos[i]
        pos[i] += h * vel[i]
        predic_pos[i] = pos[i]


@ti.kernel
def updteVelocity(h: ti.f32, pos: ti.template(), old_pos: ti.template(),
                  vel: ti.template()):
    # update velocity
    for i in pos:
        vel[i] = (pos[i] - old_pos[i]) / h


@ti.kernel
def project_constraints(mid_pos: ti.template(), tet_indices: ti.template(),
                        mass: ti.template(), lagrangian: ti.template(),
                        B: ti.template(), pos: ti.template(),
                        alpha_tilde: ti.template()):
    for i in pos:
        mid_pos[i] = pos[i]

    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = mid_pos[ia], mid_pos[ib], mid_pos[ic], mid_pos[id]
        invM0, invM1, invM2, invM3 = 1.0 / mass[ia], 1.0 / mass[
            ib], 1.0 / mass[ic], 1.0 / mass[id]
        D_s = ti.Matrix.cols([b - a, c - a, d - a])
        U, S, V = ti.svd(D_s @ B[i])
        if S[2, 2] < 0.0:  # S[2, 2] is the smallest singular value
            S[2, 2] *= -1.0
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +
                          (S[2, 2] - 1)**2)
        if constraint < 1e-12:
            continue
        g0, g1, g2, g3 = computeGradient(U, S, V, B[i])
        l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
        ) + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
        dLambda = (constraint -
                   alpha_tilde[i] * lagrangian[i]) / (l + alpha_tilde[i])
        lagrangian[i] += dLambda
        pos[ia] -= omega * invM0 * dLambda * g0
        pos[ib] -= omega * invM1 * dLambda * g1
        pos[ic] -= omega * invM2 * dLambda * g2
        pos[id] -= omega * invM3 * dLambda * g3


@ti.kernel
def collsion_response(pos: ti.template()):
    for i in pos:
        if pos[i][1] < -1.3:
            pos[i][1] = -1.3


@ti.kernel
def compute_inertial(mass: ti.template(), pos: ti.template(),
                     predict_pos: ti.template()) -> ti.f32:
    it = 0.0
    for i in pos:
        it += mass[i] * (pos[i] - predict_pos[i]).norm_sqr()
    return it * 0.5


@ti.kernel
def compute_potential_energy(pos: ti.template(), tet_indices: ti.template(),
                             B: ti.template(), alpha_tilde: ti.template()) -> ti.f32:
    pe = 0.0
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id]
        D_s = ti.Matrix.cols([b - a, c - a, d - a])
        F = D_s @ B[i]
        U, S, V = ti.svd(F)
        if S[2, 2] < 0.0:  # S[2, 2] is the smallest singular value
            S[2, 2] *= -1.0
        constraint_squared = (S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] -
                                                                    1)**2
        pe += (1.0 / alpha_tilde[i]) * constraint_squared
    return pe * 0.5


"""
    Inertial Term = \frac{1}{2} \|(x^{n+1} - \tilde{x})\|_{\mathbf{M}}
    Potential Energy = \frac{1}{2} C^{\top}(\mathbf{x})\tilde{\alpha} ^{-1} C(\mathbf{x})
    Total Energy = Kinetic Energy + Potential Energy
"""


def compute_energy(mass, pos, predict_pos, tet_indices, B, alpha_tilde):
    it = compute_inertial(mass, pos, predict_pos)
    pe = compute_potential_energy(pos, tet_indices, B, alpha_tilde)
    return it + pe, it, pe


@ti.kernel
def init_random_position(pos: ti.template(),
                         init_random_points: ti.types.ndarray()):
    for i in pos:
        pos[i] = ti.Vector([
            init_random_points[i, 0], init_random_points[i, 1],
            init_random_points[i, 2]
        ])

def log_energy(frame, filename_to_save):
    if False:
        te, it, pe = compute_energy(fmass, fpos, fpredict_pos, ftet_indices, fB, falpha_tilde)
        with open(filename_to_save, "ab") as f:
            np.savetxt(f, np.array([te]), fmt="%.4e", delimiter="\t")

def main():
    logging.getLogger().setLevel(logging.INFO)

    init_pos(fine_model_pos, fine_model_inx, fine_model_tri, fpos, fold_pos,
             fvel, fmass, ftet_indices, fB, finv_V, fdisplay_indices, fNF)
    init_pos(coarse_model_pos, coarse_model_inx, coarse_model_tri, cpos,
             cold_pos, cvel, cmass, ctet_indices, cB, cinv_V, cdisplay_indices,
             cNF)
    
    init_style = 'enlarge'

    if init_style == 'random':
        # # random init
        random_val = np.random.rand(fpos.shape[0], 3)
        fpos.from_numpy(random_val)
    elif init_style == 'enlarge':
        # init by enlarge 1.5x
        fpos.from_numpy(fine_model_pos * 1.5)

    init_alpha_tilde(falpha_tilde, finv_V)
    init_alpha_tilde(calpha_tilde, cinv_V)
    window = ti.ui.Window('3D ARAP FEM XPBD', (1300, 900), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3.5)
    camera.lookat(0, 0, 0)
    camera.fov(45)
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
    gui = window.get_gui()
    wire_frame = True
    pause = False
    show_coarse_mesh = True
    show_fine_mesh = True
    frame, max_frames = 0, 10000

    is_only_fine = False # TODO: change to False to run multigrid

    if is_only_fine:
        filename_to_save = "result/log/totalEnergy_onlyfine.txt"
    else:
        filename_to_save = "result/log/totalEnergy_mg.txt"

    if os.path.exists(filename_to_save):
        os.remove(filename_to_save)

    while window.running:
        scene.ambient_light((0.8, 0.8, 0.8))
        camera.track_user_inputs(window,
                                 movement_speed=0.03,
                                 hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        if window.is_pressed(ti.ui.ESCAPE):
            window.running = False

        if window.is_pressed(ti.ui.SPACE):
            pause = not pause

        pause = gui.checkbox("pause", pause)
        wire_frame = gui.checkbox("wireframe", wire_frame)
        show_coarse_mesh = gui.checkbox("show coarse mesh", show_coarse_mesh)
        show_fine_mesh = gui.checkbox("show fine mesh", show_fine_mesh)

        if not pause:
            info(f"######## frame {frame} ########")
            if is_only_fine:
                semiEuler(h, fpos, fpredict_pos, fold_pos, fvel, damping_coeff)
                resetLagrangian(flagrangian)
                for ite in range(only_fine_iterations):
                    log_energy(frame, filename_to_save)
                    project_constraints(fpos_mid, ftet_indices, fmass,
                                        flagrangian, fB, fpos, falpha_tilde)
                    collsion_response(fpos)
                updteVelocity(h, fpos, fold_pos, fvel)
            else:
                semiEuler(h, fpos, fpredict_pos, fold_pos, fvel, damping_coeff)
                update_coarse_mesh()
                resetLagrangian(clagrangian)
                for ite in range(coarse_iterations):
                    log_energy(frame, filename_to_save)
                    project_constraints(cpos_mid, ctet_indices, cmass,
                                        clagrangian, cB, cpos,
                                        calpha_tilde)
                    collsion_response(cpos)
                    update_fine_mesh()
                resetLagrangian(flagrangian)
                for ite in range(fine_iterations):
                    log_energy(frame, filename_to_save)
                    project_constraints(fpos_mid, ftet_indices, fmass,
                                        flagrangian, fB, fpos,
                                        falpha_tilde)
                    collsion_response(fpos)

                updteVelocity(h, fpos, fold_pos, fvel)
            frame += 1

        if frame == max_frames:
            window.running = False

        if show_fine_mesh:
            scene.mesh(fpos,
                       fdisplay_indices,
                       color=(1.0, 0.5, 0.5),
                       show_wireframe=wire_frame)

        if show_coarse_mesh:
            scene.mesh(cpos,
                       cdisplay_indices,
                       color=(0.0, 0.5, 1.0),
                       show_wireframe=wire_frame)

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()