"""
Then we use XPBD-FEM (gpu version, Jacobian solver) to simulate the deformation of 2D object
"""
import taichi as ti
from taichi.lang.ops import sqrt
from read_tet import read_tet_mesh

ti.init(arch=ti.gpu)

timeStep = 0.008  # timestep size
omega = 0.4  # SOR factor
inv_lame = 1.0e-6
MaxIte = 5
NumSteps = 10
h = timeStep / NumSteps
inv_h2 = 1.0 / h / h

DIM = 3
model_pos, model_inx, model_tri = read_tet_mesh("models/bunny1000_2000/bunny2000")

NV = len(model_pos)
NT = len(model_inx)
NF = len(model_tri)

pos = ti.Vector.field(DIM, float, NV)
predictPos = ti.Vector.field(DIM, float, NV)
oldPos = ti.Vector.field(DIM, float, NV)
vel = ti.Vector.field(DIM, float, NV)  # velocity of particles
invMass = ti.field(float, NV)  #inverse mass of particles

tet_indices = ti.Vector.field(4, int, NT)
display_indices = ti.field(ti.i32, NF * 3)

B = ti.Matrix.field(DIM, DIM, float, NT)  # D_m^{-1}
F = ti.Matrix.field(DIM, DIM, float, NT)  # deformation gradient
lagrangian = ti.field(float, NT)  # lagrangian multipliers
invVol = ti.field(float, NT)  # volume of each tet
alpha_tilde = ti.field(float, NT)  

gravity = ti.Vector([0.0, -9.8, 0.0])

@ti.kernel
def init_pos(pos_in: ti.types.ndarray(), tet_indices_in: ti.types.ndarray(),
             tri_indices_in: ti.types.ndarray()):
    for i in range(NV):
        pos[i] = ti.Vector([pos_in[i, 0], pos_in[i, 1], pos_in[i, 2]])
        oldPos[i] = pos[i]
        vel[i] = ti.Vector([0, 0, 0])
        invMass[i] = 1.0
    for i in range(NT):
        a, b, c, d = tet_indices_in[i, 0], tet_indices_in[
            i, 1], tet_indices_in[i, 2], tet_indices_in[i, 3]
        tet_indices[i] = ti.Vector([a, b, c, d])
        a, b, c, d = tet_indices[i]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        B_i_inv = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        B[i] = B_i_inv.inverse()
        invVol[i] = 6.0 / ti.abs(B_i_inv.determinant())
    for i in range(NF):
        display_indices[3 * i + 0] = tri_indices_in[i, 0]
        display_indices[3 * i + 1] = tri_indices_in[i, 1]
        display_indices[3 * i + 2] = tri_indices_in[i, 2]

@ti.kernel
def init_alpha_tilde():
    for i in range(NT):
        alpha_tilde[i] = inv_h2 * inv_lame * invVol[i]


@ti.kernel
def resetLagrangian():
    for i in range(NT):
        lagrangian[i] = 0.0


@ti.func
def vec(x):
    return ti.Vector([
        x[0, 0], x[1, 0], x[2, 0], x[0, 1], x[1, 1], x[2, 1], x[0, 2], x[1, 2],
        x[2, 2]
    ])


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.func
def computeGradient(idx, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] - 1)**2)

    # (dcdS00, dcdS11, dcdS22)
    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1, S[2, 2] - 1])
    # Compute (dFdx)^T
    neg_sum_col1 = -B[idx][0, 0] - B[idx][1, 0] - B[idx][2, 0]
    neg_sum_col2 = -B[idx][0, 1] - B[idx][1, 1] - B[idx][2, 1]
    neg_sum_col3 = -B[idx][0, 2] - B[idx][1, 2] - B[idx][2, 2]
    dFdp0T = make_matrix(neg_sum_col1, neg_sum_col2, neg_sum_col3)
    dFdp1T = make_matrix(B[idx][0, 0], B[idx][0, 1], B[idx][0, 2])
    dFdp2T = make_matrix(B[idx][1, 0], B[idx][1, 1], B[idx][1, 2])
    dFdp3T = make_matrix(B[idx][2, 0], B[idx][2, 1], B[idx][2, 2])
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
    g0 = dFdp0T @ dcdF
    g1 = dFdp1T @ dcdF
    g2 = dFdp2T @ dcdF
    g3 = dFdp3T @ dcdF
    return g0, g1, g2, g3, isSuccess


@ti.kernel
def semiEuler(h: ti.f32):
    # semi-Euler update pos & vel
    for i in range(NV):
        vel[i] = vel[i] + h * gravity
        oldPos[i] = pos[i]
        predictPos[i] = pos[i] + h * vel[i]
        pos[i] = predictPos[i]

@ti.kernel
def updteVelocity(h: ti.f32):
    # update velocity
    for i in range(NV):
        vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def project_constraints():
    for i in range(NT):
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = predictPos[ia], predictPos[ib], predictPos[ic], predictPos[id]
        invM0, invM1, invM2, invM3 = invMass[ia], invMass[ib], invMass[
            ic], invMass[id]
        D_s = ti.Matrix.cols([b - a, c - a, d - a])
        F[i] = D_s @ B[i]
        U, S, V = ti.svd(F[i])
        if S[2, 2] < 0.0:  # S[2, 2] is the smallest singular value
            S[2, 2] *= -1.0
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +
                          (S[2, 2] - 1)**2)
        g0, g1, g2, g3, isSuccess = computeGradient(i, U, S, V)
        l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
        ) + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
        dLambda = -(constraint + alpha_tilde[i] * lagrangian[i]) / (l + alpha_tilde[i])
        lagrangian[i] += dLambda
        pos[ia] += omega * invM0 * dLambda * g0
        pos[ib] += omega * invM1 * dLambda * g1
        pos[ic] += omega * invM2 * dLambda * g2
        pos[id] += omega * invM3 * dLambda * g3


@ti.kernel
def collsion_response():
    for i in range(NV):
        if pos[i][1] < -2.0:
            pos[i][1] = -2.0

@ti.kernel
def compute_kinetic_energy() -> ti.f32:
    ke = 0.0
    for i in range(NV):
        ke += 0.5 / invMass[i] * (pos[i] - predictPos[i]).norm_sqr()
    return ke

@ti.kernel
def compute_potential_energy() -> ti.f32:
    pe = 0.0
    for i in range(NT):
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id]
        D_s = ti.Matrix.cols([b - a, c - a, d - a])
        F[i] = D_s @ B[i]
        U, S, V = ti.svd(F[i])
        if S[2, 2] < 0.0:  # S[2, 2] is the smallest singular value
            S[2, 2] *= -1.0
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +
                          (S[2, 2] - 1)**2)
        pe += 0.5 * (1.0 / alpha_tilde[i]) * constraint**2
    return pe


"""
    Kinetic Eergy = \frac{1}{2} \|(x^{n+1} - \tilde{x})\|_{\mathbf{M}}
    Potential Energy = \frac{1}{2} C^{\top}(\mathbf{x})\tilde{\alpha} ^{-1} C(\mathbf{x})
    Total Energy = Kinetic Energy + Potential Energy
"""
def compute_energy():
    ke = compute_kinetic_energy()
    pe = compute_potential_energy()
    return ke + pe, ke, pe

if __name__ == "__main__":
    init_pos(pos_in=model_pos,
             tet_indices_in=model_inx,
             tri_indices_in=model_tri)
    init_alpha_tilde()
    pause = True
    window = ti.ui.Window('3D ARAP FEM XPBD', (800, 800), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3.5)
    camera.lookat(0, 0, 0)
    camera.fov(120)
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

    frame = 0
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

        if not pause:
            print(f"######## frame {frame} ########")
            for i in range(NumSteps):
                print(f"-------- substep {i} --------")
                semiEuler(h)
                resetLagrangian()
                for ite in range(MaxIte):
                    project_constraints()
                    collsion_response()
                    print(f"iteration {ite} energy {compute_energy()}")
                updteVelocity(h)
            frame += 1

        if frame == 500:
            window.running = False

        scene.mesh(pos,
                   display_indices,
                   color=(1.0, 0.5, 0.5),
                   show_wireframe=True)
        canvas.scene(scene)
        window.show()
