import taichi as ti
import numpy as np
import meshtaichi_patcher as patcher
import taichi.math as tm

ti.init()


numSubsteps = 30
dt = 1.0 / 60.0 / numSubsteps
edgeCompliance = 100.0
volumeCompliance = 1.0


# ---------------------------------------------------------------------------- #
#                                    网格和数据结构                                   #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class Mesh:
    def __init__(self, model_name="model/beam.obj"):
        self.mesh = patcher.load_mesh(model_name, relations=["FV", "VF", "FE", "EF", "EV", "VE"])

        self.mesh.verts.place(
            {"vel": ti.math.vec3, "prevPos": ti.math.vec3, "invMass": ti.f32, "pos": ti.math.vec3, "isPin": ti.u8}
        )
        self.mesh.faces.place({"restArea": ti.f32})
        self.mesh.edges.place({"restLen": ti.f32})

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())

        # 设置indices
        self.surf_show = ti.field(ti.i32, shape=len(self.mesh.faces) * 3)
        self.init_surf_indices(self.mesh, self.surf_show)

        # debug
        self.faces_debug = ti.Vector.field(3, ti.i32, len(self.mesh.faces))
        self.debug_dump_faces()

        self.init_physics()
        self.init_invMass()

        self.pins = [0, 11, 22]  # 最左边的三个点钉死
        self.pin_points()

    @ti.kernel
    def debug_dump_faces(self):
        for f in self.mesh.faces:
            for j in ti.static(range(3)):
                self.faces_debug[f.id][j] = f.verts[j].id

    @ti.kernel
    def init_surf_indices(self, mesh: ti.template(), indices: ti.template()):
        for f in mesh.faces:
            for j in ti.static(range(3)):
                indices[f.id * 3 + j] = f.verts[j].id

    @ti.kernel
    def init_physics(self):
        for f in self.mesh.faces:
            f.restArea = self.compute_area(f)
        for e in self.mesh.edges:
            e.restLen = (e.verts[0].pos - e.verts[1].pos).norm()

    @ti.kernel
    def init_invMass(self):
        density = 1.0
        for f in self.mesh.faces:
            pInvMass = 0.0
            if f.restArea > 0.0:
                pInvMass = 3.0 / (density * f.restArea)
            for v in f.verts:
                v.invMass += pInvMass

    @ti.kernel
    def pin_points(self):
        for v in self.mesh.verts:
            for i in ti.static(self.pins):
                if v.id == i:
                    v.isPin = 1
                    v.invMass = 0.0

    @ti.func
    def compute_area(self, f: ti.template()):
        x1 = f.verts[0].pos[0]
        y1 = f.verts[0].pos[1]
        x2 = f.verts[1].pos[0]
        y2 = f.verts[1].pos[1]
        x3 = f.verts[2].pos[0]
        y3 = f.verts[2].pos[1]
        return ti.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) * 0.5)


# ---------------------------------------------------------------------------- #
#                                     求解步骤                                     #
# ---------------------------------------------------------------------------- #
def substep(mesh):
    preSolve(mesh)
    solve(mesh)
    postSolve(mesh)


@ti.kernel
def preSolve(mesh: ti.template()):
    g = tm.vec3(0, -1, 0)
    for v in mesh.mesh.verts:
        if v.isPin == 1:
            continue
        v.prevPos = v.pos
        v.vel += g * dt
        v.pos += v.vel * dt
        if v.pos.y < 0.0:
            v.pos = v.prevPos
            v.pos.y = 0.0


def solve(mesh):
    solveEdge(mesh)


@ti.kernel
def solveEdge(mesh: ti.template()):
    alpha = edgeCompliance / dt / dt
    grads = tm.vec3(0, 0, 0)
    for e in mesh.mesh.edges:
        grads = e.verts[0].pos - e.verts[1].pos
        Len = grads.norm()
        grads = grads / Len
        C = Len - e.restLen

        invMass0 = e.verts[0].invMass
        invMass1 = e.verts[1].invMass
        w = invMass0 + invMass1
        s = -C / (w + alpha)

        e.verts[0].pos += grads * s * invMass0
        e.verts[1].pos += grads * (-s * invMass1)


@ti.kernel
def postSolve(mesh: ti.template()):
    for v in mesh.mesh.verts:
        v.vel = (v.pos - v.prevPos) / dt


# ---------------------------------------------------------------------------- #
#                                    可视化                                    #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    mesh = Mesh()

    window = ti.ui.Window("taichimesh", (1024, 1024))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.up(0, 1, 0)
    camera.position(0.7, 0.4, 1)
    camera.lookat(0.7, 0.4, 0)
    camera.fov(75)

    frame = 0
    paused = ti.field(int, shape=())
    paused[None] = 0
    while window.running:
        # 用下面这段代码，通过提前设置一个paused变量，我们就可以在运行的时候按空格暂停和继续了！
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
        if not paused[None]:
            substep(mesh)
            # print(f"frame: {frame}")
            frame += 1
        # 我们可以通过下面的代码来查看相机的位置和lookat，这样我们就能知道怎么调整相机的位置了
        # print("camera.curr_position",camera.curr_position)
        # print("camera.curr_lookat",camera.curr_lookat)

        # movement_speed=0.05表示移动速度，hold_key=ti.ui.RMB表示按住右键可以移动视角
        # wasdqe可以移动相机
        camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # 渲染bunny和armadillo!!
        scene.mesh(mesh.mesh.verts.pos, mesh.surf_show, color=(0.5, 0.5, 0.5), show_wireframe=True)

        scene.particles(mesh.mesh.verts.pos, radius=1e-2, color=(1, 0.5, 0.5))  # 我们也可以把点渲染出来

        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        canvas.scene(scene)

        window.show()
