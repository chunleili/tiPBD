import taichi as ti
from taichi.lang.ops import sqrt

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from engine.mesh_io import read_tetgen


class Meta:
    def __init__(self):
        self.youngs_modulus = 1.0e6
        self.poissons_ratio = 0.48
        self.lame_lambda = (
            self.youngs_modulus * self.poissons_ratio / (1 + self.poissons_ratio) / (1 - 2 * self.poissons_ratio)
        )
        self.lame_mu = self.youngs_modulus / 2 / (1 + self.poissons_ratio)
        self.inv_mu = 1.0 / self.lame_mu
        self.inv_lambda = 1.0 / self.lame_lambda
        self.gamma = 1 + self.inv_lambda / self.inv_mu  # stable neo-hookean
        self.mass_density = 1000
        self.relax_factor = 0.5
        self.gravity = ti.Vector([0.0, -9.80, 0.0])
        self.dt = 0.01
        self.inv_h2 = 1.0 / self.dt / self.dt
        self.max_iter = 30


meta = Meta()


@ti.data_oriented
class NeoHooken:
    def __init__(self) -> None:
        self.fine_model_pos, self.fine_model_inx, self.fine_model_tri = read_tetgen("data/model/cube/coarse")
        self.num_verts = len(self.fine_model_pos)
        self.num_cells = len(self.fine_model_inx)
        self.num_faces = len(self.fine_model_tri)
        self.pos = ti.Vector.field(3, float, self.num_verts)
        self.predict_pos = ti.Vector.field(3, float, self.num_verts)
        self.prev_pos = ti.Vector.field(3, float, self.num_verts)
        self.vel = ti.Vector.field(3, float, self.num_verts)  # velocity of particles
        self.mass = ti.field(float, self.num_verts)  # mass of particles
        self.inv_mass = ti.field(float, self.num_verts)  # inverse mass of particles
        self.tet_indices = ti.Vector.field(4, int, self.num_cells)
        self.B = ti.Matrix.field(3, 3, float, self.num_cells)  # D_m^{-1}
        self.lagrangian_H = ti.field(float, self.num_cells)  # lagrangian multipliers
        self.lagrangian_D = ti.field(float, self.num_cells)  # lagrangian multipliers
        self.inv_vol = ti.field(float, self.num_cells)  # volume of each tet
        self.alpha_tilde_H = ti.field(float, self.num_cells)
        self.alpha_tilde_D = ti.field(float, self.num_cells)
        self.display_indices = ti.field(ti.i32, self.num_faces * 3)
        self.init_display_indices(self.fine_model_tri, self.display_indices, self.num_faces)
        self.reset()
        self.pos_show = self.pos
        self.indices_show = self.display_indices

    def reset(self):
        self.tet_indices.from_numpy(self.fine_model_inx)
        self.pos.from_numpy(self.fine_model_pos)
        self.prev_pos.from_numpy(self.fine_model_pos)
        self.init_phsics(self.tet_indices, self.pos, self.mass, self.B, self.inv_vol, self.inv_mass)
        self.init_alpha_tilde(self.alpha_tilde_H, self.alpha_tilde_D, self.inv_vol)
        self.resetLagrangian(self.lagrangian_H, self.lagrangian_D)

    def substep(self):
        self.pre_solve(meta.dt, self.pos, self.predict_pos, self.prev_pos, self.vel)
        self.resetLagrangian(self.lagrangian_H, self.lagrangian_D)
        for ite in range(meta.max_iter):
            self.project_constraints(
                self.tet_indices,
                self.B,
                self.inv_mass,
                self.lagrangian_H,
                self.lagrangian_D,
                self.alpha_tilde_H,
                self.alpha_tilde_D,
                self.pos,
            )
            self.collsion_response(self.pos)
        self.post_solve(meta.dt, self.pos, self.prev_pos, self.vel)

    @ti.kernel
    def init_display_indices(self, tri_indices_in: ti.types.ndarray(), display_indices_out: ti.template(), NF: int):
        for i in range(NF):
            display_indices_out[3 * i + 0] = tri_indices_in[i, 0]
            display_indices_out[3 * i + 1] = tri_indices_in[i, 1]
            display_indices_out[3 * i + 2] = tri_indices_in[i, 2]

    @ti.kernel
    def init_phsics(
        self,
        tet_indices: ti.template(),
        pos: ti.template(),
        mass: ti.template(),
        B: ti.template(),
        inv_vol: ti.template(),
        inv_mass: ti.template(),
    ):
        for i in tet_indices:
            a, b, c, d = tet_indices[i][0], tet_indices[i][1], tet_indices[i][2], tet_indices[i][3]
            p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
            D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
            rest_volume = 1.0 / 6.0 * ti.abs(D_m.determinant())
            avg_mass = meta.mass_density * rest_volume / 4.0
            mass[a] += avg_mass
            mass[b] += avg_mass
            mass[c] += avg_mass
            mass[d] += avg_mass
            inv_vol[i] = 1.0 / rest_volume
            B[i] = D_m.inverse()
        for i in mass:
            inv_mass[i] = 1.0 / mass[i]

    @ti.kernel
    def init_alpha_tilde(self, alpha_tilde_H: ti.template(), alpha_tilde_D: ti.template(), inv_vol: ti.template()):
        for i in alpha_tilde_H:
            alpha_tilde_H[i] = meta.inv_h2 * meta.inv_lambda * inv_vol[i]
            alpha_tilde_D[i] = meta.inv_h2 * meta.inv_mu * inv_vol[i]

    @ti.kernel
    def resetLagrangian(self, lagrangian_H: ti.template(), lagrangian_D: ti.template()):
        for i in lagrangian_H:
            lagrangian_H[i] = 0.0
            lagrangian_D[i] = 0.0

    @ti.kernel
    def pre_solve(
        self, dt: ti.f32, pos: ti.template(), predic_pos: ti.template(), prev_pos: ti.template(), vel: ti.template()
    ):
        # semi-Euler update pos & vel
        for i in pos:
            vel[i] += dt * meta.gravity
            prev_pos[i] = pos[i]
            pos[i] += dt * vel[i]
            predic_pos[i] = pos[i]

    @ti.kernel
    def post_solve(self, dt: ti.f32, pos: ti.template(), old_pos: ti.template(), vel: ti.template()):
        # update velocity
        for i in pos:
            vel[i] = (pos[i] - old_pos[i]) / dt

    @ti.kernel
    def project_constraints(
        self,
        tet_indices: ti.template(),
        B: ti.template(),
        inv_mass: ti.template(),
        lagrangian_H: ti.template(),
        lagrangian_D: ti.template(),
        alpha_tilde_H: ti.template(),
        alpha_tilde_D: ti.template(),
        pos: ti.template(),
    ):
        for i in tet_indices:
            ia, ib, ic, id = tet_indices[i]
            a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id]
            invM0, invM1, invM2, invM3 = inv_mass[ia], inv_mass[ib], inv_mass[ic], inv_mass[id]
            D_s = ti.Matrix.cols([b - a, c - a, d - a])
            F = D_s @ B[i]

            # Constraint 1
            C_H = F.determinant() - meta.gamma
            f1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
            f2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
            f3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])

            f23 = f2.cross(f3)
            f31 = f3.cross(f1)
            f12 = f1.cross(f2)
            f = ti.Vector([f23[0], f23[1], f23[2], f31[0], f31[1], f31[2], f12[0], f12[1], f12[2]])
            dFdp1T = make_matrix(B[i][0, 0], B[i][0, 1], B[i][0, 2])
            dFdp2T = make_matrix(B[i][1, 0], B[i][1, 1], B[i][1, 2])
            dFdp3T = make_matrix(B[i][2, 0], B[i][2, 1], B[i][2, 2])

            g1 = dFdp1T @ f
            g2 = dFdp2T @ f
            g3 = dFdp3T @ f
            g0 = -g1 - g2 - g3
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr() + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
            dLambda = (-C_H - alpha_tilde_H[i] * lagrangian_H[i]) / (l + alpha_tilde_H[i])
            lagrangian_H[i] += dLambda
            pos[ia] += meta.relax_factor * invM0 * dLambda * g0
            pos[ib] += meta.relax_factor * invM1 * dLambda * g1
            pos[ic] += meta.relax_factor * invM2 * dLambda * g2
            pos[id] += meta.relax_factor * invM3 * dLambda * g3

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
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr() + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
            dLambda = (-C_D - alpha_tilde_D[i] * lagrangian_D[i]) / (l + alpha_tilde_D[i])
            lagrangian_D[i] += dLambda
            pos[ia] += meta.relax_factor * invM0 * dLambda * g0
            pos[ib] += meta.relax_factor * invM1 * dLambda * g1
            pos[ic] += meta.relax_factor * invM2 * dLambda * g2
            pos[id] += meta.relax_factor * invM3 * dLambda * g3

    @ti.kernel
    def collsion_response(self, pos: ti.template()):
        for i in pos:
            if pos[i][1] < -2:
                pos[i][1] = -2
            if pos[i][1] > 5:
                pos[i][1] = 5
            if pos[i][0] < -2:
                pos[i][0] = -2
            if pos[i][0] > 2:
                pos[i][0] = 2
            if pos[i][2] < -2:
                pos[i][2] = -2
            if pos[i][2] > 2:
                pos[i][2] = 2


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0], [0, 0, x, 0, 0, y, 0, 0, z]])
