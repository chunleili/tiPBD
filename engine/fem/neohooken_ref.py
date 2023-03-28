import taichi as ti
from taichi.lang.ops import sqrt
from read_tet import read_tet_mesh
import scipy.io as sio
import numpy as np
import sys
import os

path = os.path.abspath(os.path.join(sys.path[0])) + '/'

ti.init(arch=ti.gpu)

DIM = 3
frame, max_frames = 0, 10000
fine_max_iterations = 30

h = 0.01
inv_h2 = 1.0 / h / h
omega = 0.5  # SOR factor
gravity = ti.Vector([0.0, -9.80, 0.0])

youngs_modulus = 1.0e6
poissons_ratio = 0.48
lame_lambda = youngs_modulus * poissons_ratio / (1+poissons_ratio) / (1-2*poissons_ratio)
lame_mu = youngs_modulus / 2 / (1+poissons_ratio)
inv_mu = 1.0 / lame_mu
inv_lambda = 1.0 / lame_lambda
gamma = 1 + inv_lambda / inv_mu # stable neo-hookean
mass_density = 1000

fine_model_pos, fine_model_inx, fine_model_tri = read_tet_mesh("data/model/cube/coarse")
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
flagrangian = ti.field(float, 2 * fNT)  # lagrangian multipliers
finv_V = ti.field(float, fNT)  # volume of each tet
falpha_tilde = ti.field(float, 2 * fNT)


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
        alpha_tilde[2 * i] = inv_h2 * inv_lambda * inv_V[i]
        alpha_tilde[2 * i + 1] = inv_h2 * inv_mu * inv_V[i]


@ti.kernel
def resetLagrangian(lagrangian: ti.template()):
    for i in lagrangian:
        lagrangian[i] = 0.0


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.kernel
def semiEuler(h: ti.f32, pos: ti.template(), predic_pos: ti.template(),
              old_pos: ti.template(), vel: ti.template()):
    # semi-Euler update pos & vel
    for i in pos:
        vel[i] += h * gravity
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
        F = D_s @ B[i]

        # Constraint 1
        C_H = F.determinant() - gamma
        f1 = ti.Vector([F[0,0], F[1, 0], F[2, 0]])
        f2 = ti.Vector([F[0,1], F[1, 1], F[2, 1]])
        f3 = ti.Vector([F[0,2], F[1, 2], F[2, 2]])

        f23 = f2.cross(f3)
        f31 = f3.cross(f1)
        f12 = f1.cross(f2)
        f = ti.Vector([f23[0], f23[1], f23[2], f31[0], f31[1], f31[2], f12[0],f12[1], f12[2]])
        dFdp1T = make_matrix(B[i][0, 0], B[i][0, 1], B[i][0, 2])
        dFdp2T = make_matrix(B[i][1, 0], B[i][1, 1], B[i][1, 2])
        dFdp3T = make_matrix(B[i][2, 0], B[i][2, 1], B[i][2, 2])

        g1 = dFdp1T @ f
        g2 = dFdp2T @ f
        g3 = dFdp3T @ f
        g0 = -g1 - g2 - g3
        l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
        ) + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
        dLambda = (-C_H - alpha_tilde[2 * i] * lagrangian[2 * i]) / (l + alpha_tilde[2 * i])
        lagrangian[2 * i] += dLambda
        pos[ia] += omega * invM0 * dLambda * g0
        pos[ib] += omega * invM1 * dLambda * g1
        pos[ic] += omega * invM2 * dLambda * g2
        pos[id] += omega * invM3 * dLambda * g3

        # Constraint 2
        C_D = sqrt(f1.norm_sqr() + f2.norm_sqr() + f3.norm_sqr())
        if C_D < 1e-6:
            continue
        r_s = 1.0 / C_D
        f = ti.Vector([f1[0], f1[1], f1[2], f2[0], f2[1], f2[2], f3[0], f3[1], f3[2]])
        g1 = r_s * (dFdp1T @ f)
        g2 = r_s * (dFdp2T @ f)
        g3 = r_s * (dFdp3T @ f)
        g0 = r_s * (-g1 - g2 - g3)
        l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
        ) + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
        dLambda = (-C_D - alpha_tilde[2 * i + 1] * lagrangian[2 * i + 1]) / (l + alpha_tilde[2 * i + 1])
        lagrangian[2 * i + 1] += dLambda
        pos[ia] += omega * invM0 * dLambda * g0
        pos[ib] += omega * invM1 * dLambda * g1
        pos[ic] += omega * invM2 * dLambda * g2
        pos[id] += omega * invM3 * dLambda * g3


@ti.kernel
def collsion_response(pos: ti.template()):
    for i in pos:
        if pos[i][1] < -2:
            pos[i][1] = -2
        if pos[i][1] > 5:
            pos[i][1] = 5
        if pos[i][0] < -2:
            pos[i][0] = -2
        if pos[i][0] > 2:
            pos[i][0] = 2
        if pos[i][2] < -2:
            pos[i][2] = -2
        if pos[i][2] > 2:
            pos[i][2] = 2

def reset():
    init_pos(fine_model_pos, fine_model_inx, fine_model_tri, fpos, fold_pos,
             fvel, fmass, ftet_indices, fB, finv_V, fdisplay_indices, fNF)
    init_alpha_tilde(falpha_tilde, finv_V)
    resetLagrangian(flagrangian)


if __name__ == "__main__":
    init_pos(fine_model_pos, fine_model_inx, fine_model_tri, fpos, fold_pos,
             fvel, fmass, ftet_indices, fB, finv_V, fdisplay_indices, fNF)
    init_alpha_tilde(falpha_tilde, finv_V)
    pause = True
    window = ti.ui.Window('3D NeoHooean FEM XPBD', (1300, 900), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3.5)
    camera.lookat(0, 0, 0)
    camera.fov(100)
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
    gui = window.get_gui()
    wire_frame = True
    pause = True
    show_fine_mesh = True
    simulate_fine_mesh = True
    reset_flag = False
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
        show_fine_mesh = gui.checkbox("show fine mesh", show_fine_mesh)
        simulate_fine_mesh = gui.checkbox("simulate fine mesh",
                                          simulate_fine_mesh)
        
        reset_flag = gui.button("reset")
        if reset_flag:
            reset()

        if not pause:
            print(f"######## frame {frame} ########")
            semiEuler(h, fpos, fpredict_pos, fold_pos, fvel)
            resetLagrangian(flagrangian)
            for ite in range(fine_max_iterations):
                project_constraints(fpos_mid, ftet_indices, fmass, flagrangian,
                                    fB, fpos, falpha_tilde)
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

        canvas.scene(scene)
        window.show()
