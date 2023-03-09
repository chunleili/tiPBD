import taichi as ti
import meshtaichi_patcher as patcher
from taichi.lang.ops import sqrt
import taichi.math as tm
import numpy as np

ti.init(debug=True)

dt = 0.001  # timestep size
omega = 0.2  # SOR factor
compliance = 1.0e-3
alpha = compliance * (1.0 / dt / dt)
gravity = ti.Vector([0.0, -9.8, 0.0])
MaxIte = 2
numSubsteps = 10
rho = 1000.0

@ti.data_oriented
class Mesh:
    def __init__(self, model_name="models/toy/toy.node"):
        self.mesh = patcher.load_mesh(model_name, relations=["CV","CE","CF","VC","VE","VF","EV","EF","FE",])

        self.mesh.verts.place({ 'pos' : ti.math.vec3,
                                'vel' : ti.math.vec3,
                                'prevPos' : ti.math.vec3,
                                'invMass' : ti.f32,
                                'gradient': ti.math.vec3})
        self.mesh.cells.place({'restVol' : ti.f32,
                               'B': ti.math.mat3,
                               'F': ti.math.mat3,
                               'lagrangian': ti.f32,
                               'dLambda': ti.f32,
                               })

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())

        # 设置indices
        self.surf_show = ti.field(int, len(self.mesh.cells) * 4 * 3)
        self.init_tet_indices(self.mesh, self.surf_show)
        self.init_physics()

    @ti.kernel
    def init_tet_indices(self, mesh: ti.template(), indices: ti.template()):
        for c in mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @ti.kernel
    def init_physics(self):
        for v in self.mesh.verts:
            v.invMass = 1.0
            
        for c in self.mesh.cells:
            p0, p1, p2, p3= c.verts[0].pos, c.verts[1].pos, c.verts[2].pos, c.verts[3].pos
            Dm = tm.mat3([p1 - p0, p2 - p0, p3 - p0])
            c.B = Dm.inverse().transpose()
            c.restVol = abs(Dm.determinant()) / 6.0

mesh = Mesh()
# ---------------------------------------------------------------------------- #
#                                    核心计算步骤                                #
# ---------------------------------------------------------------------------- #

@ti.kernel
def preSolve(dt: ti.f32):
    # semi-Euler update pos & vel
    for v in mesh.mesh.verts:
        if (v.invMass != 0.0):
            v.vel = v.vel + dt * gravity
            v.prevPos = v.pos
            v.pos = v.pos + dt * v.vel

def solve():
    solveFem()


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.func
def computeGradient(B, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] - 1)**2)
    if sumSigma < 1.0e-6:
        isSuccess = False

    # (dcdS00, dcdS11, dcdS22)
    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1, S[2, 2] - 1])
    # Compute (dFdx)^T
    neg_sum_col1 = -B[0, 0] - B[1, 0] - B[2, 0]
    neg_sum_col2 = -B[0, 1] - B[1, 1] - B[2, 1]
    neg_sum_col3 = -B[0, 2] - B[1, 2] - B[2, 2]
    dFdp0T = make_matrix(neg_sum_col1, neg_sum_col2, neg_sum_col3)
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
    g0 = dFdp0T @ dcdF
    g1 = dFdp1T @ dcdF
    g2 = dFdp2T @ dcdF
    g3 = dFdp3T @ dcdF
    return g0, g1, g2, g3, isSuccess


@ti.kernel
def solveFem():
    for c in mesh.mesh.cells:
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        pos0, pos1, pos2, pos3 = p0.pos, p1.pos, p2.pos, p3.pos
        invM0, invM1, invM2, invM3 = p0.invMass, p1.invMass, p2.invMass, p3.invMass
        sumInvMass = invM0 + invM1 + invM2 + invM3

        if sumInvMass < 1.0e-6:
            print("wrong invMass function")

        D_s = ti.Matrix.cols([pos1 - pos0, pos2 - pos0, pos3 - pos0])
        c.F = D_s @ c.B

        U, S, V = ti.svd(c.F)
        if S[2, 2] < 1.0e-6:
            S[2, 2] *= -1
        
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +
                        (S[2, 2] - 1)**2)

        g0, g1, g2, g3, isSuccess = computeGradient(c.B, U, S, V)

        if isSuccess:
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
            ) + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
            c.dLambda = -(constraint + alpha * c.lagrangian) / (
                l + alpha)
            c.lagrangian = c.lagrangian + c.dLambda

            p0.gradient = g0
            p1.gradient = g1
            p2.gradient = g2
            p3.gradient = g3


@ti.kernel
def update_pos():
    for c in mesh.mesh.cells:
        for v in c.verts:
            if v.invMass != 0.0:
                v.pos += omega * v.invMass * c.dLambda * v.gradient 

    
@ti.kernel
def collsion_response():
    for v in mesh.mesh.verts:
        if v.pos[1] < -3.0:
            v.pos[1] = -3.0

@ti.kernel
def postSolve(dt: ti.f32):
    for v in mesh.mesh.verts:
        if v.invMass != 0.0:
            v.vel = (v.pos - v.prevPos) / dt


def substep():
    preSolve(dt/numSubsteps)
    mesh.mesh.cells.lagrangian.fill(0.0)
    for ite in range(MaxIte):
        solve()
        update_pos()
        collsion_response()
    postSolve(dt/numSubsteps)

def debug(field):
    field_np = field.to_numpy()
    print("---------------------")
    print("shape: ",field_np.shape)
    print("min, max: ", field_np.min(), field_np.max())
    print(field_np)
    print("---------------------")
    np.savetxt("debug_my.txt", field_np.flatten(), fmt="%.4f", delimiter="\t")
    return field_np
# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
#init the window, canvas, scene and camerea
window = ti.ui.Window("pbd", (1024, 1024),vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

#initial camera position
camera.position(0.36801182, 1.20798075, 3.1301154)
camera.lookat(0.37387108, 1.21329924, 2.13014676)
camera.fov(55)

@ti.kernel
def init_pos():
    for v in mesh.mesh.verts:
        v.pos += tm.vec3(0.5,1,0)

def main():
    # init_pos()
    paused = ti.field(int, shape=())
    paused[None] = 1


    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
            if e.key == "f":
                substep()
                # debug(mesh.mesh.verts.pos)
                print("step once")

        #do the simulation in each step
        if not paused[None]:
            for _ in range(numSubsteps):
                substep()

        #set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)


        #set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))
        
        #draw
        scene.mesh(mesh.mesh.verts.pos, indices=mesh.surf_show, color=(0.1229,0.2254,0.7207))

        #show the frame
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()