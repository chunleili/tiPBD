import taichi as ti
import meshtaichi_patcher as patcher
from taichi.lang.ops import sqrt
import taichi.math as tm
import numpy as np
import scipy.io as sio
from read_tet import read_tet_mesh

# ti.init(ti.cuda, kernel_profiler=True, debug=True)
ti.init(ti.gpu)

dt = 0.001  # timestep size
omega = 0.2  # SOR factor
compliance = 1.0e-3
# alpha = ti.field(float, ())
# alpha[None] = compliance * (1.0 / dt / dt)  # timestep related compliance
inv_lame = 1e-5
inv_h2 = 1.0 / dt / dt

gravity = ti.Vector([0.0, -9.8, 0.0])
MaxIte = 2
numSubsteps = 10

compute_energy = True
write_energy_to_file = True
potential_energy = ti.field(float, ())
kinetic_energy = ti.field(float, ())
total_energy = ti.field(float, ())
frame = ti.field(int, ())

@ti.data_oriented
class Mesh:
    def __init__(self, model_name):
        node_file = model_name + ".node"
        self.mesh = patcher.load_mesh(node_file, relations=["CV","CE","CF","VC","VE","VF","EV","EF","FE",])

        self.mesh.verts.place({ 'pos' : ti.math.vec3,
                                'vel' : ti.math.vec3,
                                'prevPos' : ti.math.vec3,
                                'predictPos' : ti.math.vec3,
                                'invMass' : ti.f32})
        self.mesh.cells.place({'restVol' : ti.f32,
                               'B': ti.math.mat3,
                               'F': ti.math.mat3,
                               'lagrangian': ti.f32,
                               'dLambda': ti.f32,
                               'grad0': ti.math.vec3,
                               'grad1': ti.math.vec3,
                               'grad2': ti.math.vec3,
                               'grad3': ti.math.vec3,
                               'alpha': ti.f32})
        #注意！这里的grad0,1,2,3是针对每个tet的四个顶点的。但是我们把他定义在cell上，而不是vert上。
        #这是因为meshtaichi中vert是唯一的（和几何点是一一对应的）。
        #也就是说多个cell共享同一个顶点时，这个顶点上的数据可能会被覆盖掉。
        #所以这里我们需要为每个tet单独存储grad0,1,2,3。

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())

        NT = len(self.mesh.cells)
        DIM = 3
        self.gradient = ti.Vector.field(DIM, float, 4 * NT)

        self.init_physics()

        # 设置显示三角面的indices
        # 自己计算surf_show
        # self.surf_show = ti.field(int, len(self.mesh.cells) * 4 * 3)
        # self.init_tet_indices(self.mesh, self.surf_show)
        # 直接读取surf_show
        surf_show_np = self.directly_import_faces(model_name + '.face')
        self.surf_show = ti.field(ti.i32, surf_show_np.shape[0] * 3)
        self.surf_show.from_numpy(surf_show_np.reshape(surf_show_np.shape[0] * 3))


    # @staticmethod
    # @ti.kernel
    # def init_tet_indices(mesh: ti.template(), indices: ti.template()):
    #     for c in mesh.cells:
    #         ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
    #         for i in ti.static(range(4)):
    #             for j in ti.static(range(3)):
    #                 indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @staticmethod
    def directly_import_faces(face_file_name):
        with open(face_file_name, 'r') as f:
            lines = f.readlines()
            NF = int(lines[0].split()[0])
            face_indices = np.zeros((NF, 3), dtype=np.int32)
            for i in range(NF):
                face_indices[i] = np.array(lines[i + 1].split()[1:-1],
                                        dtype=np.int32)
        return face_indices

    @ti.kernel
    def init_physics(self):
        for v in self.mesh.verts:
            v.invMass = 1.0

        for c in self.mesh.cells:
            p0, p1, p2, p3= c.verts[0].pos, c.verts[1].pos, c.verts[2].pos, c.verts[3].pos
            Dm = tm.mat3([p1 - p0, p2 - p0, p3 - p0])
            c.B = Dm.inverse().transpose()
            c.restVol = abs(Dm.determinant()) / 6.0
            c.alpha = inv_h2 * inv_lame * 1.0 / c.restVol

            # if c.id == 0:
            #     print("c.restVol",c.restVol)
            # pInvMass = 0.0
            # density = 1.0
            # pInvMass = 1.0/(c.restVol * density / 4.0)
            # for j in ti.static(range(4)):
            #     c.verts[j].invMass += pInvMass


mesh = Mesh(model_name="models/bunny1000_2000/bunny1000_dilate_new")


#read restriction operator
P = sio.mmread("models/bunny1000_2000/P.mtx")
fine_pos_init, fine_tet_idx, fine_tri_idx = read_tet_mesh("models/bunny1000_2000/bunny2000")
fine_pos_ti = ti.Vector.field(3, float, fine_pos_init.shape[0])
fine_num_tri = fine_tri_idx.shape[0] * 3
fine_tri_idx_ti = ti.field(ti.i32, fine_num_tri)
fine_pos_ti.from_numpy(fine_pos_init)
fine_tri_idx_ti.from_numpy(fine_tri_idx.reshape(fine_num_tri))

def update_fine_mesh():
    coarse_pos = mesh.mesh.verts.pos.to_numpy()
    fine_pos = P @ coarse_pos
    fine_pos_ti.from_numpy(fine_pos)


# ---------------------------------------------------------------------------- #
#                                    核心计算步骤                                #
# ---------------------------------------------------------------------------- #

@ti.kernel
def preSolve(dt_: ti.f32):
    # semi-Euler update pos & vel
    for v in mesh.mesh.verts:
        if (v.invMass != 0.0):
            v.vel = v.vel + dt_ * gravity
            v.prevPos = v.pos
            v.pos = v.pos + dt_ * v.vel
            v.predictPos = v.pos


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.func
def computeGradient(B, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] - 1)**2)
    # if sumSigma < 1.0e-6:
    #     isSuccess = False

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
def project_fem():
    for c in mesh.mesh.cells:
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
        c.F = D_s @ c.B
        U, S, V = ti.svd(c.F)
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)

        g0, g1, g2, g3, isSuccess = computeGradient(c.B, U, S, V)

        l = p0.invMass * g0.norm_sqr() + p1.invMass * g1.norm_sqr() + p2.invMass * g2.norm_sqr() + p3.invMass * g3.norm_sqr()
        c.dLambda = -(constraint + c.alpha * c.lagrangian) / (
            l + c.alpha)
        c.lagrangian = c.lagrangian + c.dLambda
        c.grad0, c.grad1, c.grad2, c.grad3 = g0, g1, g2, g3


@ti.kernel
def compute_potential_energy():
    potential_energy[None] = 0.0
    for c in mesh.mesh.cells:
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
        c.F = D_s @ c.B
        U, S, V = ti.svd(c.F)
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)
        potential_energy[None] += 0.5 * 1.0/c.alpha *  constraint ** 2 

@ti.kernel
def compute_kinetic_energy():
    kinetic_energy[None] = 0.0
    for v in mesh.mesh.verts:
        kinetic_energy[None] += 0.5 / v.invMass * (v.pos - v.predictPos).norm_sqr()


@ti.kernel
def update_pos():
    for c in mesh.mesh.cells:
        c.verts[0].pos += omega * c.verts[0].invMass * c.dLambda * c.grad0
        c.verts[1].pos += omega * c.verts[1].invMass * c.dLambda * c.grad1
        c.verts[2].pos += omega * c.verts[2].invMass * c.dLambda * c.grad2
        c.verts[3].pos += omega * c.verts[3].invMass * c.dLambda * c.grad3



@ti.kernel
def collsion_response():
    for v in mesh.mesh.verts:
        if v.pos[1] < -3.0:
            v.pos[1] = -3.0

@ti.kernel
def postSolve(dt_: ti.f32):
    for v in mesh.mesh.verts:
        if v.invMass != 0.0:
            v.vel = (v.pos - v.prevPos) / dt_


def substep():
    preSolve(dt/numSubsteps)
    mesh.mesh.cells.lagrangian.fill(0.0)
    for ite in range(MaxIte):
        project_fem()
        update_pos()
        collsion_response()
    postSolve(dt/numSubsteps)

    if compute_energy:
        compute_potential_energy()
        compute_kinetic_energy()
        total_energy[None] = potential_energy[None] + kinetic_energy[None]

        if write_energy_to_file and frame[None]%100==0:
            print(f"frame: {frame[None]} potential: {potential_energy[None]:.3e} kinetic: {kinetic_energy[None]:.3e} total: {total_energy[None]:.3e}")
            with open("totalEnergy.txt", "ab") as f:
                np.savetxt(f, np.array([total_energy[None]]), fmt="%.4e", delimiter="\t")
            with open("potentialEnergy.txt", "ab") as f:
                np.savetxt(f, np.array([potential_energy[None]]), fmt="%.4e", delimiter="\t")
            with open("kineticEnergy.txt", "ab") as f:
                np.savetxt(f, np.array([kinetic_energy[None]]), fmt="%.4e", delimiter="\t")

    frame[None] += 1
    


def debug(field):
    field_np = field.to_numpy()
    print("---------------------")
    print("name: ", field._name )
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
window = ti.ui.Window("pbd", (1024, 1024),vsync=False)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

#initial camera position
camera.position(-4.1016811, -1.05783201, 6.2282803)
camera.lookat(-3.50212255, -0.9375709, 5.43703646)
camera.fov(55)


def main():
    paused = ti.field(int, shape=())
    paused[None] = 1
    step=0
    show_coarse, show_fine = True, True
    
    update_fine_mesh()
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
            if e.key == "f":
                print("step: ", step)
                step+=1
                substep()
                debug(mesh.mesh.verts.pos)
                debug(mesh.mesh.cells.lagrangian)
                print("step once")

        #do the simulation in each step
        if not paused[None]:
            for _ in range(numSubsteps):
                substep()
            update_fine_mesh()

        #set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        # print("camera pos: ", camera.curr_position)
        # print("camera lookat: ", camera.curr_lookat)

        #set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        #draw
        if show_coarse:
            scene.mesh(mesh.mesh.verts.pos, indices=mesh.surf_show, color=(0.1229,0.2254,0.7207),show_wireframe=True)
        if show_fine:
            scene.mesh(fine_pos_ti,
                       fine_tri_idx_ti,
                       color=(0.5, 0.5, 1.0),
                       show_wireframe=True)

        #show the frame
        canvas.scene(scene)
        window.show()
        # ti.profiler.print_kernel_profiler_info()

if __name__ == '__main__':
    main()