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


@ti.data_oriented
class Model:
    def __init__(self, geometry_file, direct_import_faces=True):
        geometry_file_no_ext = geometry_file.split(".")[0]        
        node_file = geometry_file_no_ext + ".node"
        self.mesh = patcher.load_mesh(node_file, relations=["CV","CE","CF","VC","VE","VF","EV","EF","FE",])

        self.mesh.verts.place({ 'vel' : ti.math.vec3,
                               'pos' : ti.math.vec3,
                                'prev_pos' : ti.math.vec3,
                                'predict_pos' : ti.math.vec3,
                                'inv_mass' : ti.f32})
        self.mesh.cells.place({'inv_vol' : ti.f32,
                               'B': ti.math.mat3,
                               'lagrangian': ti.f32,
                               'fem_constraint': ti.f32,
                               'alpha': ti.f32})

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())
        self.pos = self.mesh.verts.pos
        from engine.mesh_io import scale_ti, translation_ti
        sx,sy,sz = meta.get_materials("scale", 0)
        tx,ty,tz = meta.get_materials("translation", 0)
        scale_ti(self.pos, sx,sy,sz)
        translation_ti(self.pos, tx,ty,tz)

        self.potential_energy = ti.field(float, ())
        self.inertial_energy = ti.field(float, ())
        self.total_energy = ti.field(float, ())

        self.init_physics()

        # 设置显示三角面的indices
        # 自己计算 indices_show
        if not direct_import_faces:
            self.indices_show = ti.field(int, len(self.mesh.cells) * 4 * 3)
            self.init_tet_indices(self.mesh, self.indices_show)
        # 直接读取 indices_show
        else:
            indices_show_np = self.directly_import_faces(geometry_file_no_ext + '.face')
            self.indices_show = ti.field(ti.i32, indices_show_np.shape[0] * 3)
            self.indices_show.from_numpy(indices_show_np.reshape(indices_show_np.shape[0] * 3))

    @staticmethod
    @ti.kernel
    def init_tet_indices(mesh: ti.template(), indices: ti.template()):
        for c in mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @staticmethod
    def directly_import_faces(face_file_name):
        with open(face_file_name, 'r') as f:
            lines = f.readlines()
            NF = int(lines[0].split()[0])
            face_indices = np.zeros((NF, 3), dtype=np.int32)
            for i in range(NF):
                face_indices[i] = np.array(lines[i + 1].split()[1:-1],
                                        dtype=np.int32)
        return face_indices

    @ti.kernel
    def init_physics(self):
        for v in self.mesh.verts:
            v.inv_mass = 1.0

        for c in self.mesh.cells:
            p0, p1, p2, p3= c.verts[0].pos, c.verts[1].pos, c.verts[2].pos, c.verts[3].pos
            Dm = tm.mat3([p1 - p0, p2 - p0, p3 - p0])
            c.B = Dm.inverse().transpose()
            c.inv_vol = 6.0/ abs(Dm.determinant()) 
            c.alpha = meta.inv_h2 * meta.inv_lame_lambda * c.inv_vol

@ti.data_oriented
class ARAP():
    def __init__(self):
        self.model = Model(geometry_file=meta.geometry_file)
        super().__init__()
        self.pos_show = self.model.mesh.verts.pos
        self.indices_show = self.model.indices_show
        
        if meta.get_common("use_sdf"):
            from engine.sdf import SDF
            meta.sdf_mesh_path = meta.get_sdf_meshes("geometry_file")
            self.sdf = SDF(meta.sdf_mesh_path, resolution=64, use_cache=meta.get_sdf_meshes("use_cache"))
        # from engine.visualize import vis_sdf
        # vis_sdf(self.sdf.val)


    @ti.kernel
    def project_constraints(self):
        for c in self.model.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            F = self.compute_F(p0.pos, p1.pos, p2.pos, p3.pos, c.B)
            U, S, V = ti.svd(F)
            constraint = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)
            c.fem_constraint = constraint
            g0, g1, g2, g3 = computeGradient(c.B, U, S, V)
            dlambda =  self.compute_dlambda(c, constraint, c.alpha, c.lagrangian, g0, g1, g2, g3)
            c.lagrangian += dlambda
            self.update_pos(c, dlambda, g0, g1, g2, g3)

    @ti.kernel
    def pre_solve(self, dt_: ti.f32):
        # semi-Euler update pos & vel
        for v in self.model.mesh.verts:
            # if (v.inv_mass != 0.0):
                v.vel +=  dt_ * meta.gravity
                v.prev_pos = v.pos
                v.pos += dt_ * v.vel
                v.predict_pos = v.pos
                if ti.static(meta.get_common("use_sdf")):
                    collision_response(v, self.sdf)
                collision_response_ground(v)

    @ti.func
    def update_pos(self, c, dlambda, g0, g1, g2, g3):
        c.verts[0].pos += meta.relax_factor * c.verts[0].inv_mass * dlambda * g0
        c.verts[1].pos += meta.relax_factor * c.verts[1].inv_mass * dlambda * g1
        c.verts[2].pos += meta.relax_factor * c.verts[2].inv_mass * dlambda * g2
        c.verts[3].pos += meta.relax_factor * c.verts[3].inv_mass * dlambda * g3

    @ti.kernel
    def compute_potential_energy(self):
        self.model.potential_energy[None] = 0.0
        for c in self.model.mesh.cells:
            invAlpha = meta.inv_lame_lambda * c.inv_vol
            self.model.potential_energy[None] += 0.5 * invAlpha *  c.fem_constraint ** 2 

    @ti.kernel
    def compute_inertial_energy(self):
        self.model.inertial_energy[None] = 0.0
        for v in self.model.mesh.verts:
            self.model.inertial_energy[None] += 0.5 / v.inv_mass * (v.pos - v.predict_pos).norm_sqr() * meta.inv_h2

    
    @ti.kernel
    def post_solve(self, dt_: ti.f32):
        for v in self.model.mesh.verts:
            if v.inv_mass != 0.0:
                v.vel = (v.pos - v.prev_pos) / dt_

    @ti.func
    def compute_denorminator(self, c, g0, g1, g2, g3):
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        res = p0.inv_mass * g0.norm_sqr() + p1.inv_mass * g1.norm_sqr() + p2.inv_mass * g2.norm_sqr() + p3.inv_mass * g3.norm_sqr()
        return res
    
    @ti.func
    def compute_F(self, x0,x1,x2,x3, B):
        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        res = D_s @ B
        return res

    @ti.func
    def compute_dlambda(self, c, constraint, alpha, lagrangian, g0, g1, g2, g3):
        denorminator = self.compute_denorminator(c, g0, g1, g2, g3)
        dlambda = -(constraint + alpha * lagrangian) / (denorminator + alpha)
        return dlambda

    def reset_lagrangian(self):
        self.model.mesh.cells.lagrangian.fill(0.0)

    def substep(self):
        self.pre_solve(meta.dt/meta.num_substeps)
        self.reset_lagrangian()
        for ite in range(meta.max_iter):
            self.project_constraints()
            # collsion_response(self.model.mesh.verts)
        self.post_solve(meta.dt/meta.num_substeps)

        if meta.compute_energy:
            self.compute_potential_energy()
            self.compute_inertial_energy()
            self.model.total_energy[None] = self.model.potential_energy[None] + self.model.inertial_energy[None]
            log_energy(self.model)
        meta.frame += 1


@ti.func
def collision_response_ground(v:ti.template()):
    if v.pos[1] < meta.ground.y:
        v.pos[1] = meta.ground.y

@ti.func
def collision_response(v:ti.template(), sdf):
    sdf_epsilon = 1e-4
    grid_idx = ti.Vector([v.pos.x * sdf.resolution, v.pos.y * sdf.resolution, v.pos.z * sdf.resolution], ti.i32)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]
    assert(normal.norm() == 1.0)
    if sdf_val < sdf_epsilon:
        v.pos -= sdf_val * normal


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