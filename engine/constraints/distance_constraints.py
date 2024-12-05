import taichi as ti
import numpy as np


@ti.dataclass
class DistanceConstraintAOS:
    idx: ti.i32
    c: ti.f32
    g1=ti.math.vec3([0,0,0])
    g2=ti.math.vec3([0,0,0])
    lam:ti.f32
    dlam:ti.f32
    p1:ti.i32
    p2:ti.i32
    typeid: ti.i32
    stiffness: ti.f32
    alpha:ti.f32
    rest_len: ti.f32
    dualr: ti.f32

    @ti.func
    def calc_rest_len(self,pos:ti.template()):
        self.rest_len = (pos[self.p1] - pos[self.p2]).norm()

    @ti.func
    def calc_dualr(self,delta_t):
        self.dualr = -self.c - self.alpha/delta_t/delta_t * self.lam

    @ti.func
    def calc_c(self,pos):
        self.c = (pos[self.p1] - pos[self.p2]).norm() - self.rest_len
    
    @ti.func
    def calc_grad(self,pos):
        l = (pos[self.p1] - pos[self.p2]).norm()
        if l == 0.0:
            g = ti.Vector([0.0, 0.0, 0.0])
        else:
            g = (pos[self.p1] - pos[self.p2]) / l
        self.g1 = g
        self.g2 = -g
        return g

    @ti.func
    def calc_dis(self,pos):
        dis = (pos[self.p1] - pos[self.p2])
        return dis
        


@ti.data_oriented
class DistanceConstraints():
    def __init__(self,  vert, pos, compliance=1e-8, omega=0.25):
        NCONS = vert.shape[0]

        alpha = np.empty(NCONS, dtype=np.float32)
        alpha[:] = compliance
        self.omega = omega
        self.aos =  self.init_constraints(pos,vert,alpha)

        self.inv_mass = ti.field(dtype=ti.f32, shape=pos.shape[0])
        self.inv_mass.fill(1.0)


    def set_inv_mass(self, inv_mass):
        self.inv_mass.from_numpy(inv_mass)

    def set_alpha(self, alpha):
        self.aos.alpha.from_numpy(alpha)
 
    def init_constraints(self, pos, vert, alpha):
        @ti.kernel
        def kernel(
            pos:ti.template(),
            vert:ti.template(),
            alpha: ti.types.ndarray(),
            aos: ti.template(),
        ):
            for i in range(vert.shape[0]):
                aos[i].p1, aos[i].p2 = vert[i]
                aos[i].calc_rest_len(pos)
                aos[i].alpha = alpha[i]
        
        aos = DistanceConstraintAOS.field(shape=vert.shape[0])
        kernel(pos,vert,alpha,aos)
        return aos
    

    def solve(self, pos, delta_t=3e-3, maxiter=10):
        """Public API for solving distance constraints"""
        self.aos.lam.fill(0.0)
        for i in range(maxiter):
            self.solve_one_iter(pos, delta_t)
        

    def solve_one_iter(self, pos, delta_t):
        @ti.kernel
        def _update_pos_kernel(
            inv_mass:ti.template(),
            dpos:ti.template(),
            omega:ti.f32,
            pos:ti.template(),
        ):
            for i in range(pos.shape[0]):
                if inv_mass[i] != 0.0:
                    pos[i] += omega * dpos[i]
        
        @ti.kernel
        def _solve_kernel(
            aos: ti.template(),
            inv_mass: ti.template(),
            delta_t: ti.f32,
            dpos: ti.template(),
        ):
            for i in range(dpos.shape[0]):
                dpos[i] = ti.Vector([0.0, 0.0, 0.0])

            for i in range(aos.shape[0]):
                idx0, idx1 = aos[i].p1, aos[i].p2
                invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
                constraint = aos[i].calc_c()
                g = aos[i].calc_grad()
                alpha_tilde = aos[i].alpha / delta_t / delta_t
                delta_lagrangian = -(constraint + aos[i].lam * alpha_tilde) / (invM0 + invM1 + alpha_tilde)
                aos[i].lam += delta_lagrangian

                aos[i].dualr = -(constraint + alpha_tilde * aos[i].lam)
                
                if invM0 != 0.0:
                    dpos[idx0] += invM0 * delta_lagrangian * g
                if invM1 != 0.0:
                    dpos[idx1] -= invM1 * delta_lagrangian * g

        dpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])
        _solve_kernel(self.aos, self.inv_mass, delta_t, dpos)
        _update_pos_kernel(self.inv_mass, dpos, self.omega, pos)
