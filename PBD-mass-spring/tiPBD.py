from mesh_data import bunnyMesh
import taichi as ti
import numpy as np
import meshtaichi_patcher as patcher
import taichi.math as tm
ti.init()

numSubsteps = 10
dt = 1.0 / 60.0 / numSubsteps
edgeCompliance = 100.0
volumeCompliance = 0.0

@ti.data_oriented
class Mesh:
    def __init__(self):

        #读取网格
        self.numParticles = len(bunnyMesh['verts']) // 3
        self.numEdges = len(bunnyMesh['tetEdgeIds']) // 2
        self.numTets = len(bunnyMesh['tetIds']) // 4
        self.numSurfs = len(bunnyMesh['tetSurfaceTriIds']) // 3

        pos_np = np.array(bunnyMesh['verts'], dtype=float)
        tet_np = np.array(bunnyMesh['tetIds'], dtype=int)
        edge_np = np.array(bunnyMesh['tetEdgeIds'], dtype=int)
        surf_np = np.array(bunnyMesh['tetSurfaceTriIds'], dtype=int)

        pos_np = pos_np.reshape((-1,3))
        tet_np = tet_np.reshape((-1,4))
        edge_np = edge_np.reshape((-1,2))
        surf_np = surf_np.reshape((-1,3))

        #定义太极数据结构
        self.mesh = ti.TetMesh()

        self.pos = ti.Vector.field(3, float, self.numParticles)
        self.tet = ti.Vector.field(4, int, self.numTets)
        self.edge = ti.Vector.field(2, int, self.numEdges)
        self.surf = ti.Vector.field(3, int, self.numSurfs)
        
        self.pos.from_numpy(pos_np)
        self.tet.from_numpy(tet_np)
        self.edge.from_numpy(edge_np)
        self.surf.from_numpy(surf_np)

        self.restLen = ti.field(float, self.numEdges)
        self.restVol = ti.field(float, self.numTets)
        self.invMass = ti.field(float, self.numParticles)

        self.init_physics()
        self.init_invMass()

        self.prevPos = ti.Vector.field(3, float, self.numParticles)
        self.vel = ti.Vector.field(3, float, self.numParticles)
        self.surf_show = ti.field(int, self.numSurfs * 3)
        self.surf_show.from_numpy(surf_np.flatten())



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
        return res
    
mesh = Mesh()
# mesh.init_physics()
# mesh.init_invMass()
# ---------------------------------------------------------------------------- #
#                                    核心计算步骤                                    #
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
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)

@ti.kernel
def init_pos():
    for i in range(mesh.numParticles):
        mesh.pos[i] += tm.vec3(0.5,1,0)

def main():
    init_pos()
    while window.running:
        #do the simulation in each step
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
        scene.mesh(mesh.pos, indices=mesh.surf_show, color=(1,1,0))

        #show the frame
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()