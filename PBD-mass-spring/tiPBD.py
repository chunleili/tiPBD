from mesh_data import bunnyMesh
import taichi as ti
import numpy as np
import meshtaichi_patcher as patcher
import taichi.math as tm
ti.init()

numSubsteps = 30
dt = 1.0 / 60.0 / numSubsteps
edgeCompliance = 100.0
volumeCompliance = 1.0

@ti.data_oriented
class Mesh:
    def __init__(self, model_name="models/toy.node"):
        self.mesh = patcher.load_mesh(model_name, relations=["CV","CE","CF","VC","VE","VF","EV","EF","FE",])

        self.numParticles = len(self.mesh.verts)
        self.numTets = len(self.mesh.cells)
        self.numEdges = len(self.mesh.edges)
        self.numSurfs = len(self.mesh.faces)
        self.mesh.verts.pos = ti.Vector.field(3, float, self.numParticles)
        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())

        # 设置indices
        self.surf_show = ti.field(int, len(self.mesh.cells) * 4 * 3)
        self.init_tet_indices(self.mesh, self.surf_show)

        self.pos = self.mesh.verts.pos
        self.tet = ti.Vector.field(4, int, self.numTets)
        self.edge = ti.Vector.field(2, int, self.numEdges)
        self.dump()

        self.restLen = ti.field(float, self.numEdges)
        self.restVol = ti.field(float, self.numTets)
        self.invMass = ti.field(float, self.numParticles)

        self.init_physics()
        self.init_invMass()

        self.prevPos = ti.Vector.field(3, float, self.numParticles)
        self.vel = ti.Vector.field(3, float, self.numParticles)

    @ti.kernel
    def dump(self):
        for c in self.mesh.cells:
            for j in ti.static(range(4)):
                self.tet[c.id][j] = c.verts[j].id
        for e in self.mesh.edges:
            for j in ti.static(range(2)):
                self.edge[e.id][j] = e.verts[j].id


    @ti.kernel
    def init_tet_indices(self, mesh: ti.template(), indices: ti.template()):
        for c in mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @ti.kernel
    def init_physics(self):
        for i in self.restVol:
            self.restVol[i] = self.tetVolume(i)
        for i in self.restLen:
            self.restLen[i] = (self.pos[self.edge[i][0]] - self.pos[self.edge[i][1]]).norm()

    @ti.kernel
    def init_invMass(self):
        for i in range(self.numTets):
            pInvMass = 0.0
            if self.restVol[i] > 0.0:
                pInvMass = 1.0 / (self.restVol[i] / 4.0)
            for j in ti.static(range(4)):
                self.invMass[self.tet[i][j]] += pInvMass
    
    @ti.func
    def tetVolume(self,i):
        id = tm.ivec4(-1,-1,-1,-1)
        for j in ti.static(range(4)):
            id[j] = self.tet[i][j]
        temp = (self.pos[id[1]] - self.pos[id[0]]).cross(self.pos[id[2]] - self.pos[id[0]])
        res = temp.dot(self.pos[id[3]] - self.pos[id[0]])
        res *= 1.0/6.0
        return ti.abs(res)
    
mesh = Mesh()
# ---------------------------------------------------------------------------- #
#                                    核心计算步骤                                #
# ---------------------------------------------------------------------------- #

@ti.kernel
def preSolve():
    g = tm.vec3(0, -1, 0)
    for i in mesh.pos:
        mesh.prevPos[i] = mesh.pos[i]
        mesh.vel[i] += g * dt 
        mesh.pos[i] += mesh.vel[i] * dt
        if mesh.pos[i].y < 0.0:
            mesh.pos[i] = mesh.prevPos[i]
            mesh.pos[i].y = 0.0

def solve():
    solveEdge()
    solveVolume()

@ti.kernel
def solveEdge():
    alpha = edgeCompliance / dt / dt
    grads = tm.vec3(0,0,0)
    for i in range(mesh.numEdges):
        id0 = mesh.edge[i][0]
        id1 = mesh.edge[i][1]

        grads = mesh.pos[id0] - mesh.pos[id1]
        Len = grads.norm()
        grads = grads / Len
        C =  Len - mesh.restLen[i]
        w = mesh.invMass[id0] + mesh.invMass[id1]
        s = -C / (w + alpha)

        mesh.pos[id0] += grads *   s * mesh.invMass[id0]
        mesh.pos[id1] += grads * (-s * mesh.invMass[id1])


@ti.kernel
def solveVolume():
    alpha = volumeCompliance / dt / dt
    grads = [tm.vec3(0,0,0), tm.vec3(0,0,0), tm.vec3(0,0,0), tm.vec3(0,0,0)]
    
    for i in range(mesh.numTets):
        id = tm.ivec4(-1,-1,-1,-1)
        for j in ti.static(range(4)):
            id[j] = mesh.tet[i][j]
        grads[0] = (mesh.pos[id[3]] - mesh.pos[id[1]]).cross(mesh.pos[id[2]] - mesh.pos[id[1]])
        grads[1] = (mesh.pos[id[2]] - mesh.pos[id[0]]).cross(mesh.pos[id[3]] - mesh.pos[id[0]])
        grads[2] = (mesh.pos[id[3]] - mesh.pos[id[0]]).cross(mesh.pos[id[1]] - mesh.pos[id[0]])
        grads[3] = (mesh.pos[id[1]] - mesh.pos[id[0]]).cross(mesh.pos[id[2]] - mesh.pos[id[0]])

        w = 0.0
        for j in ti.static(range(4)):
            w += mesh.invMass[id[j]] * (grads[j].norm())**2

        vol = mesh.tetVolume(i)
        C = (vol - mesh.restVol[i]) * 6.0
        s = -C /(w + alpha)
        
        for j in ti.static(range(4)):
            mesh.pos[mesh.tet[i][j]] += grads[j] * s * mesh.invMass[id[j]]
        
@ti.kernel
def postSolve():
    for i in mesh.pos:
        mesh.vel[i] = (mesh.pos[i] - mesh.prevPos[i]) / dt
    
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
    for i in range(mesh.numParticles):
        mesh.pos[i] += tm.vec3(0.5,1,0)

def main():
    init_pos()
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
        # scene.particles(mesh.pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(mesh.pos, indices=mesh.surf_show, color=(0.1229,0.2254,0.7207))

        #show the frame
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()