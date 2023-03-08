from mesh_data import bunnyMesh
import taichi as ti
import numpy as np
ti.init()

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

# ---------------------------------------------------------------------------- #
#                      precompute the restLen and restVol                      #
# ---------------------------------------------------------------------------- #
import taichi.math as tm

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