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
    for t in tet:
        p0 = tet[t][0]
        p1 = tet[t][1]
        p2 = tet[t][2]
        p3 = tet[t][3]
        x0 = pos[p0]
        x1 = pos[p1] - x0
        x2 = pos[p2] - x0
        x3 = pos[p3] - x0

        D_s = ti.Matrix.cols([x1 , x2 , x3 ])
        F  = D_s @ B[t]
        # S = F.transpose() @ F 
        #S是3x3的矩阵，S[i,j]是个标量
        # compute_grad_s(F, B[t], grad_s)

        # 先计算stretch约束
        for i,j in ti.static([0,0],[0,1],[2,2]):
            fi = get_column(F,i)
            fj = get_column(F,j)
            ci = get_column(B[t],i)
            cj = get_column(B[t],j)
            gradSij = fj.outer_product(ci) + fi.outer_product(cj)
            gradSij_p1 = get_column(gradSij,0)
            gradSij_p2 = get_column(gradSij,1)
            gradSij_p3 = get_column(gradSij,2)
            denorminator =  inv_mass[p1] * gradSij_p1.norm_sqr() + inv_mass[p2] * gradSij_p2.norm_sqr() + inv_mass[p3] * gradSij_p3.norm_sqr()
            Sij = fi.dot(fj)
            lambda_ = (Sij - 1.0) / denorminator 
            gradSij_p0 = -gradSij_p1 - gradSij_p2 - gradSij_p3
            dx0[p0] += -lambda_*inv_mass[p0]* gradSij_p0
            dx1[p1] += -lambda_*inv_mass[p1]* gradSij_p1
            dx2[p2] += -lambda_*inv_mass[p2]* gradSij_p2
            dx3[p3] += -lambda_*inv_mass[p3]* gradSij_p3

        # 再计算shear约束
        for i,j in ti.static([0,1],[0,2],[1,2]):
            fi = get_column(F,i)
            fj = get_column(F,j)
            ci = get_column(B[t],i)
            cj = get_column(B[t],j)
            gradSij = fj.outer_product(ci) + fi.outer_product(cj)
            gradSij_p1 = get_column(gradSij,0)
            gradSij_p2 = get_column(gradSij,1)
            gradSij_p3 = get_column(gradSij,2)
            denorminator =  inv_mass[p1] * gradSij_p1.norm_sqr() + inv_mass[p2] * gradSij_p2.norm_sqr() + inv_mass[p3] * gradSij_p3.norm_sqr()
            Sij = fi.dot(fj)
            lambda_ = Sij / denorminator 
            gradSij_p0 = -gradSij_p1 - gradSij_p2 - gradSij_p3
            dx0[p0] += -lambda_*inv_mass[p0]* gradSij_p0
            dx1[p1] += -lambda_*inv_mass[p1]* gradSij_p1
            dx2[p2] += -lambda_*inv_mass[p2]* gradSij_p2
            dx3[p3] += -lambda_*inv_mass[p3]* gradSij_p3


    for t in tet:
        p0 = tet[t][0]
        p1 = tet[t][1]
        p2 = tet[t][2]
        p3 = tet[t][3]
        pos[p0] += meta.relax_factor * dx0[p0]
        pos[p1] += meta.relax_factor * dx1[p1]
        pos[p2] += meta.relax_factor * dx2[p2]
        pos[p3] += meta.relax_factor * dx3[p3]

# bool PositionBasedDynamics::solve_StrainTetraConstraint(
# 	const Vector3r &p0, Real invMass0, 
# 	const Vector3r &p1, Real invMass1,
# 	const Vector3r &p2, Real invMass2,
# 	const Vector3r &p3, Real invMass3,
# 	const Matrix3r &invRestMat,
# 	const Vector3r &stretchStiffness,	
# 	const Vector3r &shearStiffness,	
# 	const bool normalizeStretch,
# 	const bool normalizeShear,
# 	Vector3r &corr0, Vector3r &corr1, Vector3r &corr2, Vector3r &corr3)
# {
# 	corr0.setZero();
# 	corr1.setZero();
# 	corr2.setZero();
# 	corr3.setZero();

# 	Vector3r c[3];
# 	c[0] = invRestMat.col(0);
# 	c[1] = invRestMat.col(1);
# 	c[2] = invRestMat.col(2);

# 	for (int i = 0; i < 3; i++) {
# 		for (int j = 0; j <= i; j++) {

# 			Matrix3r P;
# // 			P.col(0) = p1 - p0;		// Jacobi
# // 			P.col(1) = p2 - p0;
# // 			P.col(2) = p3 - p0;

# 			P.col(0) = (p1 + corr1) - (p0 + corr0);		// Gauss - Seidel
# 			P.col(1) = (p2 + corr2) - (p0 + corr0);
# 			P.col(2) = (p3 + corr3) - (p0 + corr0);

# 			Vector3r fi = P * c[i];
# 			Vector3r fj = P * c[j];

# 			Real Sij = fi.dot(fj);

# 			Real wi,wj,s1,s3;
# 			if (normalizeShear && i != j) {
# 				wi = fi.norm();
# 				wj = fj.norm();
# 				s1 = static_cast<Real>(1.0) / (wi*wj);
# 				s3 = s1 * s1 * s1;
# 			}

# 			Vector3r d[4];
# 			d[0] = Vector3r(0.0, 0.0, 0.0);

# 			for (int k = 0; k < 3; k++) {
# 				d[k+1] = fj * invRestMat(k,i) + fi * invRestMat(k,j);

# 				if (normalizeShear && i != j) {
# 					d[k+1] = s1 * d[k+1] - Sij*s3 * (wj*wj * fi*invRestMat(k,i) + wi*wi * fj*invRestMat(k,j));
# 				}

# 				d[0] -= d[k+1];
# 			}

# 			if (normalizeShear && i != j)
# 				Sij *= s1;

# 			Real lambda = 
# 				invMass0 * d[0].squaredNorm() +
# 				invMass1 * d[1].squaredNorm() +
# 				invMass2 * d[2].squaredNorm() +
# 				invMass3 * d[3].squaredNorm();

# 			if (fabs(lambda) < eps)		// foo: threshold should be scale dependent
# 				continue;

# 			if (i == j) {	// diagonal, stretch
# 				if (normalizeStretch)  {
# 					Real s = sqrt(Sij);
# 					lambda = static_cast<Real>(2.0) * s * (s - static_cast<Real>(1.0)) / lambda * stretchStiffness[i];
# 				}
# 				else {
# 					lambda = (Sij - static_cast<Real>(1.0)) / lambda * stretchStiffness[i];
# 				}
# 			}
# 			else {		// off diagonal, shear
# 				lambda = Sij / lambda * shearStiffness[i + j - 1];
# 			}

# 			corr0 -= lambda * invMass0 * d[0];
# 			corr1 -= lambda * invMass1 * d[1];
# 			corr2 -= lambda * invMass2 * d[2];
# 			corr3 -= lambda * invMass3 * d[3];
# 		}
# 	}
# 	return true;
# }

@ti.func
def get_columns(m:mat3):
    return ti.Vector([m[0, 0], m[1, 0], m[2, 0]]), ti.Vector([m[0, 1], m[1, 1], m[2, 1]]), ti.Vector([m[0, 2], m[1, 2], m[2, 2]])

@ti.func
def get_column(m:mat3, i:int):
    return vec3(m[0, i], m[1, i], m[2, i])

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
    def substep(self):
        substep_()