import taichi as ti
import taichi.math as tm
import numpy as np
import logging

from data.model.mass_spring_volumetric_meshdata import bunnyMesh
from engine.metadata import meta

# ti.init()

# if meta.args.arch != "cpu":
#     logging.error("This example only supports CPU arch for now. We will force switch to CPU arch")
#     meta.args.init_args["arch"] = "cpu"
#     ti.init(**meta.args.init_args)


numParticles = len(bunnyMesh['verts']) // 3
numEdges = len(bunnyMesh['tetEdgeIds']) // 2
numTets = len(bunnyMesh['tetIds']) // 4
numSurfs = len(bunnyMesh['tetSurfaceTriIds']) // 3

pos_np = np.array(bunnyMesh['verts'], dtype=float)
tet_np = np.array(bunnyMesh['tetIds'], dtype=int)
edge_np = np.array(bunnyMesh['tetEdgeIds'], dtype=int)
surf_np = np.array(bunnyMesh['tetSurfaceTriIds'], dtype=int)

pos_np = pos_np.reshape((-1,3))
tet_np = tet_np.reshape((-1,4))
edge_np = edge_np.reshape((-1,2))
surf_np = surf_np.reshape((-1,3))

pos = ti.Vector.field(3, float, numParticles)
tet = ti.Vector.field(4, int, numTets)
edge = ti.Vector.field(2, int, numEdges)
surf = ti.Vector.field(3, int, numSurfs)

pos.from_numpy(pos_np)
tet.from_numpy(tet_np)
edge.from_numpy(edge_np)
surf.from_numpy(surf_np)

def transform():
    from engine.mesh_io import scale_ti, translation_ti
    sx, sy, sz = meta.get_materials("scale")
    tx, ty, tz = meta.get_materials("translation")
    scale_ti(pos, sx, sy, sz)
    translation_ti(pos, tx, ty, tz)

transform()

# if meta.get_materials("use_sdf"):
from engine.sdf import SDF
meta.sdf_mesh_path = meta.get_sdf_meshes("geometry_file")
sdf = SDF(meta.sdf_mesh_path, resolution=64, use_cache=meta.get_sdf_meshes("use_cache"))

# ---------------------------------------------------------------------------- #
#                      precompute the restLen and restVol                      #
# ---------------------------------------------------------------------------- #

restVol = ti.field(float, numTets)
restLen = ti.field(float, numEdges)
invMass = ti.field(float, numParticles)

@ti.func
def tetVolume(i):
    id = tm.ivec4(-1,-1,-1,-1)
    for j in ti.static(range(4)):
        id[j] = tet[i][j]
    temp = (pos[id[1]] - pos[id[0]]).cross(pos[id[2]] - pos[id[0]])
    res = temp.dot(pos[id[3]] - pos[id[0]])
    res *= 1.0/6.0
    return res

@ti.kernel
def init_physics():
    for i in restVol:
        restVol[i] = tetVolume(i)
    for i in restLen:
        restLen[i] = (pos[edge[i][0]] - pos[edge[i][1]]).norm()

@ti.kernel
def init_invMass():
    for i in range(numTets):
        pInvMass = 0.0
        if restVol[i] > 0.0:
            pInvMass = 1.0 / (restVol[i] / 4.0)
        for j in ti.static(range(4)):
            invMass[tet[i][j]] += pInvMass

init_physics()
init_invMass()

# ti.init()

# numSubsteps = 10
# edgeCompliance = 100.0
# volumeCompliance = 0.0
# dt = 1.0 / 60.0 / numSubsteps

numSubsteps = meta.get_common("num_substeps",10)
dt = 1.0 / 60.0 / numSubsteps
edgeCompliance = meta.get_materials("edge_compliance",100.0)
volumeCompliance = meta.get_materials("volume_compliance",0.0)

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
        # collision_response(pos[i], sdf)
        if pos[i].y < 0.0:
            # pos[i] = prevPos[i]
            pos[i].y = 0.0

@ti.func
def collision_response(pos:ti.template(), sdf):
    sdf_epsilon = 1e-4
    grid_idx = ti.Vector([pos.x * sdf.resolution, pos.y * sdf.resolution, pos.z * sdf.resolution], ti.i32)
    grid_idx = ti.math.clamp(grid_idx, 0, sdf.resolution - 1)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]
    assert 1 - 1e-4 < normal.norm() < 1 + 1e-4, f"sdf normal norm is not one: {normal.norm()}" 
    if sdf_val < sdf_epsilon:
        pos -= sdf_val * normal

def solve():
    solveEdge()
    solveVolume()


dpos_edge_0 = ti.Vector.field(3, float, numEdges)
dpos_edge_1 = ti.Vector.field(3, float, numEdges)

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

        # pos[id0] += grads *   s * invMass[id0]
        # pos[id1] += grads * (-s * invMass[id1])
        dpos_edge_0[i] = grads *   s * invMass[id0]
        dpos_edge_1[i] = grads * (-s * invMass[id1])


dpos_vol_0 = ti.Vector.field(3, float, numTets)
dpos_vol_1 = ti.Vector.field(3, float, numTets)
dpos_vol_2 = ti.Vector.field(3, float, numTets)
dpos_vol_3 = ti.Vector.field(3, float, numTets)

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
        
        # for j in ti.static(range(4)):
        #     pos[tet[i][j]] += grads[j] * s * invMass[id[j]]
        dpos_vol_0[i] = grads[0] * s * invMass[id[0]]
        dpos_vol_1[i] = grads[1] * s * invMass[id[1]]
        dpos_vol_2[i] = grads[2] * s * invMass[id[2]]
        dpos_vol_3[i] = grads[3] * s * invMass[id[3]]


@ti.kernel
def apply_dpos():
    for i in pos:
        pos[i] += dpos_edge_0[i] + dpos_edge_1[i] + dpos_vol_0[i] + dpos_vol_1[i] + dpos_vol_2[i] + dpos_vol_3[i]


@ti.kernel
def postSolve():
    for i in pos:
        vel[i] = (pos[i] - prevPos[i]) / dt
    
def substep_():
    preSolve()
    solve()
    postSolve()


# @ti.kernel
# def init_pos():
#     for i in range(numParticles):
#         pos[i] += tm.vec3(0.5,1,0)

# init_pos()


class MassSpring():
    def __init__(self) -> None:
        self.pos_show = pos
        self.indices_show = surf_show
        if meta.get_common("use_sdf", False):
            self.sdf = sdf
    def substep(self):
        substep_()


# window = ti.ui.Window("pbd", (1024, 1024),vsync=False)
# canvas = window.get_canvas()
# scene = ti.ui.Scene()
# camera = ti.ui.make_camera()

# #initial camera position
# camera.position(0.5, 1.0, 1.95)
# camera.lookat(0.5, 0.3, 0.5)
# camera.fov(55)
# def main():
#     while window.running:
#         #do the simulation in each step
#         for _ in range(numSubsteps):
#             substep()

#         #set the camera, you can move around by pressing 'wasdeq'
#         camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
#         scene.set_camera(camera)

#         #set the light
#         scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
#         scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
#         scene.ambient_light((0.5, 0.5, 0.5))
        
#         #draw
#         # scene.particles(pos, radius=0.02, color=(0, 1, 1))
#         scene.mesh(pos, indices=surf_show, color=(1,1,0))

#         #show the frame
#         canvas.scene(scene)
#         window.show()

# if __name__ == '__main__':
#     main()