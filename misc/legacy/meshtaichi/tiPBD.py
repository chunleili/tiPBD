import taichi as ti
import meshtaichi_patcher as patcher
import taichi.math as tm

ti.init()

numSubsteps = 30
dt = 1.0 / 60.0 / numSubsteps
edgeCompliance = 100.0
volumeCompliance = 1.0


@ti.data_oriented
class Mesh:
    def __init__(self, model_name="model/toy/toy.node"):
        self.mesh = patcher.load_mesh(
            model_name,
            relations=[
                "CV",
                "CE",
                "CF",
                "VC",
                "VE",
                "VF",
                "EV",
                "EF",
                "FE",
            ],
        )

        self.mesh.verts.place(
            {
                "vel": ti.math.vec3,
                "prevPos": ti.math.vec3,
                "invMass": ti.f32,
                "pos": ti.math.vec3,
            }
        )
        self.mesh.cells.place({"restVol": ti.f32})
        self.mesh.edges.place({"restLen": ti.f32})

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())

        # 设置indices
        self.surf_show = ti.field(int, len(self.mesh.cells) * 4 * 3)
        self.init_tet_indices(self.mesh, self.surf_show)

        self.init_physics()
        self.init_invMass()

    @ti.kernel
    def init_tet_indices(self, mesh: ti.template(), indices: ti.template()):
        for c in mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @ti.kernel
    def init_physics(self):
        for c in self.mesh.cells:
            c.restVol = self.tetVolume(c)
        for e in self.mesh.edges:
            e.restLen = (e.verts[0].pos - e.verts[1].pos).norm()

    @ti.kernel
    def init_invMass(self):
        for c in self.mesh.cells:
            pInvMass = 0.0
            if c.restVol > 0.0:
                pInvMass = 1.0 / (c.restVol / 4.0)
            for v in c.verts:
                v.invMass += pInvMass

    @ti.func
    def tetVolume(self, c: ti.template()):
        temp = (c.verts[1].pos - c.verts[0].pos).cross(c.verts[2].pos - c.verts[0].pos)
        res = temp.dot(c.verts[3].pos - c.verts[0].pos)
        res *= 1.0 / 6.0
        return ti.abs(res)


mesh = Mesh()
# ---------------------------------------------------------------------------- #
#                                    核心计算步骤                                #
# ---------------------------------------------------------------------------- #


@ti.kernel
def preSolve():
    g = tm.vec3(0, -1, 0)
    for v in mesh.mesh.verts:
        v.prevPos = v.pos
        v.vel += g * dt
        v.pos += v.vel * dt
        if v.pos.y < 0.0:
            v.pos = v.prevPos
            v.pos.y = 0.0


def solve():
    solveEdge()
    solveVolume()


@ti.kernel
def solveEdge():
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
def solveVolume():
    alpha = volumeCompliance / dt / dt
    grads = [tm.vec3(0, 0, 0), tm.vec3(0, 0, 0), tm.vec3(0, 0, 0), tm.vec3(0, 0, 0)]

    for c in mesh.mesh.cells:
        grads[0] = (c.verts[3].pos - c.verts[1].pos).cross(c.verts[2].pos - c.verts[1].pos)
        grads[1] = (c.verts[2].pos - c.verts[0].pos).cross(c.verts[3].pos - c.verts[0].pos)
        grads[2] = (c.verts[3].pos - c.verts[0].pos).cross(c.verts[1].pos - c.verts[0].pos)
        grads[3] = (c.verts[1].pos - c.verts[0].pos).cross(c.verts[2].pos - c.verts[0].pos)

        w = 0.0
        for j in ti.static(range(4)):
            w += c.verts[j].invMass * (grads[j].norm()) ** 2

        vol = mesh.tetVolume(c)
        C = (vol - c.restVol) * 6.0
        s = -C / (w + alpha)

        for j in ti.static(range(4)):
            c.verts[j].pos += grads[j] * s * c.verts[j].invMass


@ti.kernel
def postSolve():
    for v in mesh.mesh.verts:
        v.vel = (v.pos - v.prevPos) / dt


def substep():
    preSolve()
    solve()
    postSolve()


# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
# init the window, canvas, scene and camerea
window = ti.ui.Window("pbd", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

# initial camera position
camera.position(0.36801182, 1.20798075, 3.1301154)
camera.lookat(0.37387108, 1.21329924, 2.13014676)
camera.fov(45)


@ti.kernel
def init_pos():
    for v in mesh.mesh.verts:
        v.pos += tm.vec3(0.5, 1, 0)


def main():
    init_pos()
    paused = ti.field(int, shape=())
    paused[None] = 0
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
            if e.key == "f":
                substep()
                print("step once")

        # do the simulation in each step
        if not paused[None]:
            for _ in range(numSubsteps):
                substep()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw
        scene.mesh(mesh.mesh.verts.pos, indices=mesh.surf_show, color=(0.1229, 0.2254, 0.7207))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
