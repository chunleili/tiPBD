import taichi as ti
import meshtaichi_patcher as patcher
import numpy as np
import taichi.math as tm
from engine.metadata import meta
from engine.log import log_energy
from taichi.math import mat3, vec3

arr_t = ti.types.ndarray()

meta.dt = meta.common["dt"]
meta.relax_factor = 0.005
meta.gravity = ti.Vector(meta.common["gravity"])
meta.ground = ti.Vector(meta.common["ground"])
meta.max_iter = meta.common["max_iter"]
meta.num_substeps = meta.common["num_substeps"]
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

dx0 = ti.Vector.field(3, float, pos.shape[0])
dx1 = ti.Vector.field(3, float, pos.shape[0])
dx2 = ti.Vector.field(3, float, pos.shape[0])
dx3 = ti.Vector.field(3, float, pos.shape[0])

grad_s = ti.Matrix.field(3,3,dtype=float, shape=(3,3))

@ti.kernel
def project_constraints():
    normalize_shear = False
    normalize_stretch = False
    stretch_stiffness = vec3(1.)
    shear_stiffness = vec3(1.)
    for t in tet:
        p0 = tet[t][0]
        p1 = tet[t][1]
        p2 = tet[t][2]
        p3 = tet[t][3]
        x0 = pos[p0]
        x1 = pos[p1] - x0
        x2 = pos[p2] - x0
        x3 = pos[p3] - x0
        inv_mass0 = inv_mass[p0]
        inv_mass1 = inv_mass[p1]
        inv_mass2 = inv_mass[p2]
        inv_mass3 = inv_mass[p3]
        solve_StrainTetraConstraint(x0, x1, x2, x3, inv_mass0, inv_mass1, inv_mass2, inv_mass3, B[t], stretch_stiffness, shear_stiffness, normalize_shear, normalize_stretch, dx0[p0], dx1[p1], dx2[p2], dx3[p3])


    #Jacobian iteration
    for t in tet:
        p0 = tet[t][0]
        p1 = tet[t][1]
        p2 = tet[t][2]
        p3 = tet[t][3]
        if (inv_mass[p0] != 0.0):
            if t ==0:
                print(dx0[p0], dx1[p1], dx2[p2], dx3[p3])
            pos[p0] += dx0[p0]
        if (inv_mass[p1] != 0.0):
            pos[p1] += dx1[p1]
        if (inv_mass[p2] != 0.0):
            pos[p2] += dx2[p2]
        if (inv_mass[p3] != 0.0):
            pos[p3] += dx3[p3]

@ti.func
def solve_StrainTetraConstraint(x0, x1, x2, x3, inv_mass0, inv_mass1, inv_mass2, inv_mass3, B, stretch_stiffness, shear_stiffness, normalize_shear, normalize_stretch, dx0:ti.template(), dx1:ti.template(), dx2:ti.template(), dx3:ti.template()):

    dx0, dx1, dx2, dx3 = vec3(0), vec3(0), vec3(0), vec3(0)
    c=B
    for i in (range(3)):
        for j in ti.static(range(3)):
            if j <= i:
                P = mat3(0)
                P[:,0] = x1 - x0
                P[:,1] = x2 - x0
                P[:,2] = x3 - x0
                
                fi = vec3(P@c[i,:])
                fj = vec3(P@c[j,:])
                
                Sij = fi.dot(fj)
                wi,wj,s1,s3=0.,0.,0.,0.
                if normalize_shear and i!=j:
                    wi = fi.norm()
                    wj = fj.norm()
                    s1 = 1.0/(wi*wj)
                    s3 = s1**3

                d = [vec3(0),vec3(0),vec3(0),vec3(0)]
                for k in ti.static(range(3)):
                    d[k+1] = fj * B[k,i] + fi * B[k,j]

                    if normalize_shear and i!=j:
                        d[k+1] += s1 * d[k+1] - Sij * s3 * (wj * wj * fi * B[k,i] + wi * wi * fj * B[k,j])
                    d[0] -= d[k+1]
                
                if normalize_shear and i!=j:
                    Sij *= s1

                lam = inv_mass0 * d[0].norm_sqr() + inv_mass1 * d[1].norm_sqr() + inv_mass2 * d[2].norm_sqr() + inv_mass3 * d[3].norm_sqr()

                if(i==j):
                    if(normalize_stretch):
                        s = ti.sqrt(Sij)
                        lam = 2. * s * (s-1.) / lam * stretch_stiffness[i]
                    else:
                        lam = ( Sij - 1. ) / lam * stretch_stiffness[i]
                else:
                    lam = Sij / lam * shear_stiffness[i+j-1]

                dx0 += -lam * inv_mass0 * d[0]
                dx1 += -lam * inv_mass1 * d[1]
                dx2 += -lam * inv_mass2 * d[2]
                dx3 += -lam * inv_mass3 * d[3]



def clear_dx():
    dx0.fill(vec3(0))
    dx1.fill(vec3(0))
    dx2.fill(vec3(0))
    dx3.fill(vec3(0))

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
        meta.iter = ite+1
        clear_dx()
        project_constraints()
    post_solve(meta.dt/meta.num_substeps)



@ti.data_oriented
class StrainBasedDynamics():
    def __init__(self):
        self.pos_show = pos
        self.indices_show = face
        init_physics()
    def substep(self):
        substep_()