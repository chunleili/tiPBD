import taichi as ti
import meshtaichi_patcher as patcher
import numpy as np
import taichi.math as tm
from engine.metadata import meta
from engine.log import log_energy

meta.dt = meta.common["dt"]
meta.relax_factor = meta.common["relax_factor"]
meta.gravity = ti.Vector(meta.common["gravity"])
meta.ground = ti.Vector(meta.common["ground"])
meta.max_iter = meta.common["max_iter"]
meta.num_substeps = meta.common["num_substeps"]
meta.use_multigrid = meta.common["use_multigrid"]
meta.show_coarse, meta.show_fine = meta.common["show_coarse"], meta.common["show_fine"]
meta.compute_energy = meta.common["compute_energy"]
meta.use_log = meta.common["use_log"]
meta.paused = meta.common["initial_pause"]
meta.inv_h2 = 1.0 / meta.dt / meta.dt

meta.geometry_file = meta.materials[0]["geometry_file"] # only support one solid for now
meta.lame_lambda = meta.materials[0]["lame_lambda"]
meta.inv_lame_lambda = 1.0/meta.lame_lambda
meta.geometry_file_noext = meta.geometry_file.split(".")[0]

from engine.mesh_io import read_tetgen
pos_np, tet_np, face_np = read_tetgen(meta.geometry_file_noext)
face_np = face_np.flatten()
pos = ti.Vector.field(3, dtype=ti.f32, shape=pos_np.shape[0])
tet = ti.Vector.field(4, dtype=ti.i32, shape=tet_np.shape[0])
face = ti.field(dtype=ti.i32, shape=face_np.shape[0])
pos.from_numpy(pos_np)
tet.from_numpy(tet_np)
face.from_numpy(face_np)


from engine.mesh_io import scale_ti, translation_ti
sx,sy,sz = meta.get_materials("scale", 0)
tx,ty,tz = meta.get_materials("translation", 0)
scale_ti(pos, sx,sy,sz)
translation_ti(pos, tx,ty,tz)


potential_energy = ti.field(float, ())
inertial_energy = ti.field(float, ())
total_energy = ti.field(float, ())


inv_mass = ti.field(float, pos.shape[0])
B = ti.Matrix.field(3, 3, float, tet.shape[0])
alpha = ti.field(float, tet.shape[0])
lagrangian = ti.field(float, tet.shape[0])
vel = ti.Vector.field(3, float, pos.shape[0])
prev_pos = ti.Vector.field(3, float, pos.shape[0])

@ti.kernel
def init_physics():
    for i in inv_mass:
        inv_mass[i] = 1.0

    for t in tet:
        # p0, p1, p2, p3= c.verts[0].pos, c.verts[1].pos, c.verts[2].pos, c.verts[3].pos
        p0 = pos[tet[t][0]]
        p1 = pos[tet[t][1]]
        p2 = pos[tet[t][2]]
        p3 = pos[tet[t][3]]

        Dm = tm.mat3([p1 - p0, p2 - p0, p3 - p0])
        B[t] = Dm.inverse().transpose()
        inv_vol = 6.0/ abs(Dm.determinant()) 
        alpha[t] = meta.inv_h2 * meta.inv_lame_lambda * inv_vol


        
if meta.get_common("use_sdf"):
    from engine.sdf import SDF
    meta.sdf_mesh_path = meta.get_sdf_meshes("geometry_file")
    sdf = SDF(meta.sdf_mesh_path, resolution=64, use_cache=meta.get_sdf_meshes("use_cache"))

if meta.get_common("initialize_random"):
    random_val = np.random.rand(pos.shape[0], 3)
    pos.from_numpy(random_val)

@ti.kernel
def project_constraints():
    for t in tet:
        p0 = tet[t][0]
        p1 = tet[t][1]
        p2 = tet[t][2]
        p3 = tet[t][3]

        x0 = pos[p0]
        x1 = pos[p1]
        x2 = pos[p2]
        x3 = pos[p3]

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F  = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)
        g0, g1, g2, g3 = computeGradient(B[t], U, S, V)
        denorminator =  inv_mass[p0] * g0.norm_sqr() + inv_mass[p1] * g1.norm_sqr() + inv_mass[p2] * g2.norm_sqr() + inv_mass[p3] * g3.norm_sqr()
        dlambda = -(constraint + alpha[t] * lagrangian[t]) / (denorminator + alpha[t])

        lagrangian[t] += dlambda
        
        pos[p0] += meta.relax_factor * inv_mass[p0] * dlambda * g0
        pos[p1] += meta.relax_factor * inv_mass[p1] * dlambda * g1
        pos[p2] += meta.relax_factor * inv_mass[p2] * dlambda * g2
        pos[p3] += meta.relax_factor * inv_mass[p3] * dlambda * g3


@ti.kernel
def pre_solve(dt:float):
    g = tm.vec3(0, -1, 0)
    for i in pos:
        prev_pos[i] = pos[i]
        vel[i] += g * dt 
        pos[i] += vel[i] * dt
        if pos[i].y < 0.0:
            pos[i].y = 0.0
        if ti.static(meta.get_common("use_sdf")):
            collision_response_sdf(pos[i], sdf)

@ti.func
def collision_response_sdf(pos:ti.template(), sdf):
    sdf_epsilon = 1e-4
    grid_idx = ti.Vector([pos.x * sdf.resolution, pos.y * sdf.resolution, pos.z * sdf.resolution], ti.i32)
    grid_idx = ti.math.clamp(grid_idx, 0, sdf.resolution - 1)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]
    assert 1 - 1e-4 < normal.norm() < 1 + 1e-4, f"sdf normal norm is not one: {normal.norm()}" 
    if sdf_val < sdf_epsilon:
        pos -= sdf_val * normal

@ti.kernel
def post_solve(dt:float):
    for i in pos:
        vel[i] = (pos[i] - prev_pos[i]) / dt


def substep_():
    pre_solve(meta.dt/meta.num_substeps)
    lagrangian.fill(0.0)
    for ite in range(meta.max_iter):
        project_constraints()
    post_solve(meta.dt/meta.num_substeps)


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])

@ti.func
def computeGradient(B, U, S, V):
    sumSigma = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] - 1)**2)

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
    return g0, g1, g2, g3


# #read restriction operator
# P = sio.mmread("data/model/bunny1k2k/P.mtx")
# fine_mesh = Mesh(geometry_file="data/model/bunny1k2k/bunny2k.node", direct_import_faces=True)

# def coarse_to_fine():
#     coarse_pos = mesh.mesh.verts.pos.to_numpy()
#     fine_pos = P @ coarse_pos
#     fine_mesh.mesh.verts.pos.from_numpy(fine_pos)



@ti.data_oriented
class ARAP():
    def __init__(self):
        self.pos_show = pos
        self.indices_show = face
        self.sdf = sdf
    def substep(self):
        substep_()