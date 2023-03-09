import taichi as ti
import meshtaichi_patcher as patcher
import taichi.math as tm
ti.init()

h = 0.001  # timestep size
omega = 0.2  # SOR factor
compliance = 1.0e-3
alpha = compliance * (1.0 / h / h)
gravity = ti.Vector([0.0, -9.8, 0.0])
MaxIte = 2
NumSteps = 10
rho = 1000.0

@ti.data_oriented
class Mesh:
    def __init__(self, model_name="models/toy/toy.node"):
        self.mesh = patcher.load_mesh(model_name, relations=["CV","CE","CF","VC","VE","VF","EV","EF","FE",])

        self.mesh.verts.place({ 'pos' : ti.math.vec3,
                                'vel' : ti.math.vec3,
                                'prevPos' : ti.math.vec3,
                                'invMass' : ti.f32})
        self.mesh.cells.place({'restVol' : ti.f32,
                               'B': ti.math.mat3,
                               'F': ti.math.mat3,
                               'lagrangian': ti.f32,
                               'dLambda': ti.f32,
                               'gradient': ti.f32})

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
            c.B = Dm.inverse()
            c.restVol = abs(Dm.determinant()) / 6.0

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
    solveFem()

@ti.kernel
def solveFem():
    pass

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
#init the window, canvas, scene and camerea
window = ti.ui.Window("pbd", (1024, 1024),vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

#initial camera position
camera.position(0.36801182, 1.20798075, 3.1301154)
camera.lookat(0.37387108, 1.21329924, 2.13014676)
camera.fov(55)

@ti.kernel
def init_pos():
    for v in mesh.mesh.verts:
        v.pos += tm.vec3(0.5,1,0)

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