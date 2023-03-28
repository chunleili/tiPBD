import taichi as ti
import taichi.math as tm
from engine.fem.mesh import Mesh
from engine.metadata import meta
from engine.log import log_energy

@ti.data_oriented
class FemBase:
    def __init__(self, model_name):
        self.mesh = Mesh(model_name=model_name)

    @ti.kernel
    def pre_solve(self, dt_: ti.f32):
        # semi-Euler update pos & vel
        for v in self.mesh.mesh.verts:
            if (v.invMass != 0.0):
                v.vel = v.vel + dt_ * meta.gravity
                v.prevPos = v.pos
                v.pos = v.pos + dt_ * v.vel
                v.predictPos = v.pos

    @ti.kernel
    def project_constraints():
        # to be implemented in derived class
        pass

    @ti.kernel
    def update_pos(self):
        for c in self.mesh.mesh.cells:
            c.verts[0].pos += meta.relax_factor * c.verts[0].invMass * c.dLambda * c.grad0
            c.verts[1].pos += meta.relax_factor * c.verts[1].invMass * c.dLambda * c.grad1
            c.verts[2].pos += meta.relax_factor * c.verts[2].invMass * c.dLambda * c.grad2
            c.verts[3].pos += meta.relax_factor * c.verts[3].invMass * c.dLambda * c.grad3

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
            self.mesh.inertial_energy[None] += 0.5 / v.invMass * (v.pos - v.predictPos).norm_sqr() * meta.inv_h2
    
    @ti.kernel
    def collsion_response(self):
        for v in self.mesh.mesh.verts:
            if v.pos[1] < meta.ground.y:
                v.pos[1] = meta.ground.y
    
    @ti.kernel
    def post_solve(self, dt_: ti.f32):
        for v in self.mesh.mesh.verts:
            if v.invMass != 0.0:
                v.vel = (v.pos - v.prevPos) / dt_

    def substep(self):
        self.pre_solve(meta.dt/meta.num_substeps)
        self.mesh.mesh.cells.lagrangian.fill(0.0)
        for ite in range(meta.max_iter):
            self.project_constraints()
            self.update_pos()
            self.collsion_response()
        self.post_solve(meta.dt/meta.num_substeps)

        if meta.compute_energy:
            self.compute_potential_energy()
            self.compute_inertial_energy()
            self.mesh.total_energy[None] = self.mesh.potential_energy[None] + self.mesh.inertial_energy[None]
            log_energy(self.mesh)
        meta.frame += 1