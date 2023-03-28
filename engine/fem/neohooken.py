import taichi as ti
from taichi.lang.ops import sqrt
import taichi.math as tm
from engine.metadata import meta
from engine.fem.fem_base import FemBase


from read_tet import read_tet_mesh
import sys
import os

path = os.path.abspath(os.path.join(sys.path[0])) + '/'

ti.init(arch=ti.gpu)

DIM = 3
frame, max_frames = 0, 10000
fine_max_iterations = 30

h = 0.01
inv_h2 = 1.0 / h / h
omega = 0.5  # SOR factor
gravity = ti.Vector([0.0, -9.80, 0.0])

youngs_modulus = 1.0e6
poissons_ratio = 0.48
lame_lambda = youngs_modulus * poissons_ratio / (1+poissons_ratio) / (1-2*poissons_ratio)
lame_mu = youngs_modulus / 2 / (1+poissons_ratio)
inv_mu = 1.0 / lame_mu
inv_lambda = 1.0 / lame_lambda
gamma = 1 + inv_lambda / inv_mu # stable neo-hookean
mass_density = 1000

fine_model_pos, fine_model_inx, fine_model_tri = read_tet_mesh("data/model/cube/coarse")
fNV = len(fine_model_pos)
fNT = len(fine_model_inx)
fNF = len(fine_model_tri)
fpos = ti.Vector.field(DIM, float, fNV)
fpos_mid = ti.Vector.field(DIM, float, fNV)
fpredict_pos = ti.Vector.field(DIM, float, fNV)
fold_pos = ti.Vector.field(DIM, float, fNV)
fvel = ti.Vector.field(DIM, float, fNV)  # velocity of particles
fmass = ti.field(float, fNV)  # mass of particles
ftet_indices = ti.Vector.field(4, int, fNT)
fdisplay_indices = ti.field(ti.i32, fNF * 3)
fB = ti.Matrix.field(DIM, DIM, float, fNT)  # D_m^{-1}
flagrangian = ti.field(float, 2 * fNT)  # lagrangian multipliers
finv_V = ti.field(float, fNT)  # volume of each tet
falpha_tilde = ti.field(float, 2 * fNT)


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])



@ti.data_oriented
class ARAP(FemBase):
    def __init__(self):
        super().__init__()
    
    def update_pos(self):
        pass # do nothing

    @ti.kernel
    def project_constraints(self):
        for c in self.mesh.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
            F = D_s @ c.B

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
            
            l = p0.inv_mass * g0.norm_sqr() + p1.inv_mass * g1.norm_sqr() + p2.inv_mass * g2.norm_sqr() + p3.inv_mass * g3.norm_sqr()
            dlambda = -(C_H + c.alpha * c.lagrangian) / (l + c.alpha)
            c.lagrangian += dlambda

            c.verts[0].pos += meta.relax_factor * c.verts[0].inv_mass * dlambda * g0
            c.verts[1].pos += meta.relax_factor * c.verts[1].inv_mass * dlambda * g1
            c.verts[2].pos += meta.relax_factor * c.verts[2].inv_mass * dlambda * g2
            c.verts[3].pos += meta.relax_factor * c.verts[3].inv_mass * dlambda * g3


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
            l = p0.inv_mass * g0.norm_sqr() + p1.inv_mass * g1.norm_sqr() + p2.inv_mass * g2.norm_sqr() + p3.inv_mass * g3.norm_sqr()
            dlambda = (-C_D - c.alpha * c.lagrangian) / (l + c.alpha)
            c.lagrangian += dlambda
            c.verts[0].pos += meta.relax_factor * c.verts[0].inv_mass * dlambda * g0
            c.verts[1].pos += meta.relax_factor * c.verts[1].inv_mass * dlambda * g1
            c.verts[2].pos += meta.relax_factor * c.verts[2].inv_mass * dlambda * g2
            c.verts[3].pos += meta.relax_factor * c.verts[3].inv_mass * dlambda * g3
