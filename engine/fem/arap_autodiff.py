import taichi as ti
from engine.fem.fem_base import FemBase


@ti.data_oriented
class ARAP_autodiff(FemBase):
    def __init__(self):
        super().__init__()

        self.cell_size = len(self.mesh.mesh.cells)
        self.constraint = ti.field(ti.f32, shape=self.cell_size, needs_grad=True)
        self.x0 = ti.Vector.field(3, ti.f32, shape=self.cell_size, needs_grad=True)
        self.x1 = ti.Vector.field(3, ti.f32, shape=self.cell_size, needs_grad=True)
        self.x2 = ti.Vector.field(3, ti.f32, shape=self.cell_size, needs_grad=True)
        self.x3 = ti.Vector.field(3, ti.f32, shape=self.cell_size, needs_grad=True)
        self.B_ex = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_size)
        self.B_ex.copy_from(self.mesh.mesh.cells.B)

    @ti.kernel
    def dump(self):
        for c in self.mesh.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            self.x0[c.id], self.x1[c.id], self.x2[c.id], self.x3[c.id] = p0.pos, p1.pos, p2.pos, p3.pos

    @ti.kernel
    def compute_constraint(self):
        for cid in range(self.cell_size):
            F = self.compute_F(self.x0[cid], self.x1[cid], self.x2[cid], self.x3[cid], self.B_ex[cid])
            U, S, V = ti.svd(F)
            self.constraint[cid] = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)


    def project_constraints(self):
        self.dump()
        self.constraint.grad.fill(1.0)
        self.x0.grad.fill(0.0)
        self.x1.grad.fill(0.0)
        self.x2.grad.fill(0.0)
        self.x3.grad.fill(0.0)
        self.compute_constraint.grad()
        g0,g1,g2,g3 = self.x0.grad, self.x1.grad, self.x2.grad, self.x3.grad
        self.apply_gradient(self.constraint, g0,g1,g2,g3)

    @ti.kernel
    def apply_gradient(self,constraint:ti.template(), g0:ti.template(), g1:ti.template(), g2:ti.template(), g3:ti.template()):
        for c in self.mesh.mesh.cells:
            dlambda =  self.compute_dlambda(c, constraint[c.id], c.alpha, c.lagrangian, g0[c.id],g1[c.id],g2[c.id],g3[c.id])
            c.lagrangian += dlambda
            self.update_pos(c, dlambda, g0[c.id],g1[c.id],g2[c.id],g3[c.id])