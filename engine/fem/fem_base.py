import taichi as ti
import taichi.math as tm
from engine.fem.mesh import Mesh
from engine.metadata import meta
from engine.log import log_energy

@ti.data_oriented
class FemBase:
    def __init__(self):
        self.mesh = Mesh(geometry_file=meta.geometry_file)

    @ti.kernel
    def pre_solve(self, dt_: ti.f32):
        # semi-Euler update pos & vel
        for v in self.mesh.mesh.verts:
            if (v.inv_mass != 0.0):
                v.vel = v.vel + dt_ * meta.gravity
                v.prev_pos = v.pos
                v.pos = v.pos + dt_ * v.vel
                v.predict_pos = v.pos

    @ti.kernel
    def project_constraints():
        # to be implemented in derived class
        pass

    @ti.func
    def update_pos(self, c, dlambda, g0, g1, g2, g3):
        c.verts[0].pos += meta.relax_factor * c.verts[0].inv_mass * dlambda * g0
        c.verts[1].pos += meta.relax_factor * c.verts[1].inv_mass * dlambda * g1
        c.verts[2].pos += meta.relax_factor * c.verts[2].inv_mass * dlambda * g2
        c.verts[3].pos += meta.relax_factor * c.verts[3].inv_mass * dlambda * g3

    @ti.kernel
    def compute_potential_energy(self):
        self.mesh.potential_energy[None] = 0.0
        for c in self.mesh.mesh.cells:
            invAlpha = meta.inv_lame_lambda * c.inv_vol
            self.mesh.potential_energy[None] += 0.5 * invAlpha *  c.fem_constraint ** 2 

    @ti.kernel
    def compute_inertial_energy(self):
        self.mesh.inertial_energy[None] = 0.0
        for v in self.mesh.mesh.verts:
            self.mesh.inertial_energy[None] += 0.5 / v.inv_mass * (v.pos - v.predict_pos).norm_sqr() * meta.inv_h2
    
    @ti.kernel
    def collsion_response(self):
        for v in self.mesh.mesh.verts:
            if v.pos[1] < meta.ground.y:
                v.pos[1] = meta.ground.y
    
    @ti.kernel
    def post_solve(self, dt_: ti.f32):
        for v in self.mesh.mesh.verts:
            if v.inv_mass != 0.0:
                v.vel = (v.pos - v.prev_pos) / dt_

    @ti.func
    def compute_denorminator(self, c, g0, g1, g2, g3):
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        res = p0.inv_mass * g0.norm_sqr() + p1.inv_mass * g1.norm_sqr() + p2.inv_mass * g2.norm_sqr() + p3.inv_mass * g3.norm_sqr()
        return res
    
    @ti.func
    def compute_F(self, c, B):
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
        res = D_s @ B
        return res

    @ti.func
    def compute_dlambda(self, c, constraint, alpha, lagrangian, g0, g1, g2, g3):
        denorminator = self.compute_denorminator(c, g0, g1, g2, g3)
        dlambda = -(constraint + alpha * lagrangian) / (denorminator + alpha)
        return dlambda

    def substep(self):
        self.pre_solve(meta.dt/meta.num_substeps)
        self.mesh.mesh.cells.lagrangian.fill(0.0)
        for ite in range(meta.max_iter):
            self.project_constraints()
            self.collsion_response()
        self.post_solve(meta.dt/meta.num_substeps)

        if meta.compute_energy:
            self.compute_potential_energy()
            self.compute_inertial_energy()
            self.mesh.total_energy[None] = self.mesh.potential_energy[None] + self.mesh.inertial_energy[None]
            log_energy(self.mesh)
        meta.frame += 1