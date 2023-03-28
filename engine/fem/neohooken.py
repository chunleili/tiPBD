import taichi as ti
from taichi.lang.ops import sqrt
import taichi.math as tm
from engine.metadata import meta
from engine.fem.fem_base import FemBase


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.data_oriented
class NeoHooken(FemBase):
    def __init__(self):
        super().__init__()

    @ti.kernel
    def project_constraints(self):
        for c in self.mesh.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            
            F = self.compute_F(c, c.B)

            # Constraint 1 
            C_H = F.determinant() - meta.gamma
            f1 = ti.Vector([F[0,0], F[1, 0], F[2, 0]])
            f2 = ti.Vector([F[0,1], F[1, 1], F[2, 1]])
            f3 = ti.Vector([F[0,2], F[1, 2], F[2, 2]])

            f23 = f2.cross(f3)
            f31 = f3.cross(f1)
            f12 = f1.cross(f2)
            f = ti.Vector([f23[0], f23[1], f23[2], f31[0], f31[1], f31[2], f12[0],f12[1], f12[2]])
            dFdp1T = make_matrix(c.B[0, 0], c.B[0, 1], c.B[0, 2])
            dFdp2T = make_matrix(c.B[1, 0], c.B[1, 1], c.B[1, 2])
            dFdp3T = make_matrix(c.B[2, 0], c.B[2, 1], c.B[2, 2])

            g1 = dFdp1T @ f
            g2 = dFdp2T @ f
            g3 = dFdp3T @ f
            g0 = -g1 - g2 - g3
            
            dlambda =  self.compute_dlambda(c, C_H, c.alpha, c.lagrangian, g0, g1, g2, g3)
            c.lagrangian += dlambda
            self.update_pos(c, dlambda, g0, g1, g2, g3)


            # Constraint 2
            C_D = sqrt(f1.norm_sqr() + f2.norm_sqr() + f3.norm_sqr())
            if C_D < 1e-6:
                continue
            r_s = 1.0 / C_D
            f = ti.Vector([f1[0], f1[1], f1[2], f2[0], f2[1], f2[2], f3[0], f3[1], f3[2]])
            g1 = r_s * (dFdp1T @ f)
            g2 = r_s * (dFdp2T @ f)
            g3 = r_s * (dFdp3T @ f)
            g0 = r_s * (-g1 - g2 - g3)


            dlambda = self.compute_dlambda(c, C_D, c.alpha, c.lagrangian, g0, g1, g2, g3)
            c.lagrangian += dlambda
            self.update_pos(c, dlambda, g0, g1, g2, g3)
