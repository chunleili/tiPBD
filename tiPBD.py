import taichi as ti
import taichi.math as tm
from tiReadMesh import *
# ti.init()

numSubsteps = 10
dt = 1.0 / 60.0 / numSubsteps
edgeCompliance = 100.0
volumeCompliance = 0.0

prevPos = ti.Vector.field(3, float, numParticles)
vel = ti.Vector.field(3, float, numParticles)

surf_show = ti.field(int, numSurfs * 3)
surf_show.from_numpy(surf_np.flatten())

@ti.kernel
def preSolve():
    g = tm.vec3(0, -1, 0)
    for i in pos:
        prevPos[i] = pos[i]
        vel[i] += g * dt 
        pos[i] += vel[i] * dt
        if pos[i].y < 0.0:
            pos[i] = prevPos[i]
            pos[i].y = 0.0

def solve():
    solveEdge()
    solveVolume()

@ti.kernel
def solveEdge():
    alpha = edgeCompliance / dt / dt
    grads = tm.vec3(0,0,0)
    for i in range(numEdges):
        id0 = edge[i][0]
        id1 = edge[i][1]

        grads = pos[id0] - pos[id1]
        Len = grads.norm()
        grads = grads / Len
        C =  Len - restLen[i]
        w = invMass[id0] + invMass[id1]
        s = -C / (w + alpha)

        pos[id0] += grads *   s * invMass[id0]
        pos[id1] += grads * (-s * invMass[id1])


@ti.kernel
def solveVolume():
    alpha = volumeCompliance / dt / dt
    grads = [tm.vec3(0,0,0), tm.vec3(0,0,0), tm.vec3(0,0,0), tm.vec3(0,0,0)]
    
    for i in range(numTets):
        id = tm.ivec4(-1,-1,-1,-1)
        for j in ti.static(range(4)):
            id[j] = tet[i][j]
        grads[0] = (pos[id[3]] - pos[id[1]]).cross(pos[id[2]] - pos[id[1]])
        grads[1] = (pos[id[2]] - pos[id[0]]).cross(pos[id[3]] - pos[id[0]])
        grads[2] = (pos[id[3]] - pos[id[0]]).cross(pos[id[1]] - pos[id[0]])
        grads[3] = (pos[id[1]] - pos[id[0]]).cross(pos[id[2]] - pos[id[0]])

        w = 0.0
        for j in ti.static(range(4)):
            w += invMass[id[j]] * (grads[j].norm())**2

        vol = tetVolume(i)
        C = (vol - restVol[i]) * 6.0
        s = -C /(w + alpha)
        
        for j in ti.static(range(4)):
            pos[tet[i][j]] += grads[j] * s * invMass[id[j]]
        
@ti.kernel
def postSolve():
    for i in pos:
        vel[i] = (pos[i] - prevPos[i]) / dt
    
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
    for i in range(numParticles):
        pos[i] += tm.vec3(0.5,1,0)

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
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(pos, indices=surf_show, color=(1,1,0))

        #show the frame
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()