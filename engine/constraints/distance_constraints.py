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
        g = ti.Vector([0.0, 0.0, 0.0])
        if l == 0.0:
            g = ti.Vector([0.0, 0.0, 0.0])
        else:
            g = (pos[self.p1] - pos[self.p2]) / l
        return g

    @ti.func
    def calc_dis(self,pos):
        dis = (pos[self.p1] - pos[self.p2])
        return dis
        


@ti.data_oriented
class DistanceConstraintsAlongEdge():
    """Distance Along Edge"""
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
            pos: ti.template(),
            delta_t: ti.f32,
            dpos: ti.template(),
        ):
            for i in range(dpos.shape[0]):
                dpos[i] = ti.Vector([0.0, 0.0, 0.0])

            for i in range(aos.shape[0]):
                idx0, idx1 = aos[i].p1, aos[i].p2
                invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
                l = (pos[idx0] - pos[idx1]).norm()
                if l == 0.0:
                    continue
                constraint = l - aos[i].rest_len
                g = (pos[idx0] - pos[idx1]) / l
                alpha_tilde = aos[i].alpha / delta_t / delta_t
                delta_lagrangian = -(constraint + aos[i].lam * alpha_tilde) / (invM0 + invM1 + alpha_tilde)
                aos[i].lam += delta_lagrangian

                aos[i].dualr = -(constraint + alpha_tilde * aos[i].lam)
                
                if invM0 != 0.0:
                    dpos[idx0] += invM0 * delta_lagrangian * g
                if invM1 != 0.0:
                    dpos[idx1] -= invM1 * delta_lagrangian * g


        dpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])
        _solve_kernel(self.aos, self.inv_mass, pos, delta_t, dpos)
        _update_pos_kernel(self.inv_mass, dpos, self.omega, pos)


@ti.data_oriented
class DistanceConstraintsAttach():
    """Attach to target geometry"""
    def __init__(self,  pts, target_pt, pos, target_pos, compliance=1e-9, omega=0.25):
        NCONS = pts.shape[0]

        alpha = np.empty(NCONS, dtype=np.float32)
        alpha[:] = compliance
        self.omega = omega
        self.aos =  self.init_constraints(pts, target_pt, pos, target_pos, alpha)

        self.inv_mass = ti.field(dtype=ti.f32, shape=pos.shape[0])
        self.inv_mass.fill(1.0)
        self.target_inv_mass = ti.field(dtype=ti.f32, shape=pos.shape[0])
        self.target_inv_mass.fill(1.0)

        self.pts = pts
        self.target_pt = target_pt

    def set_inv_mass(self, inv_mass):
        self.inv_mass.from_numpy(inv_mass)

    def set_target_inv_mass(self, target_inv_mass):
        self.target_inv_mass.from_numpy(target_inv_mass)

    def set_alpha(self, alpha):
        self.aos.alpha.from_numpy(alpha)

    def set_rest_len(self, rest_len):
        self.aos.rest_len.from_numpy(rest_len)

    def set_p1(self, target_pt):
        self.aos.p1.from_numpy(target_pt)

    def set_p2(self, pts):
        self.aos.p1.from_numpy(pts)
 
    def init_constraints(self, pts, target_pt, pos, target_pos, alpha):
        @ti.kernel
        def kernel(
            pts:ti.template(),
            target_pt:ti.template(),
            pos:ti.template(),
            target_pos:ti.template(),
            alpha: ti.types.ndarray(),
            aos: ti.template(),
        ):
            for i in range(pts.shape[0]):
                aos[i].p1, aos[i].p2 = target_pt[i], pts[i]
                aos[i].rest_len = (target_pos[target_pt[i]] - pos[pts[i]]).norm()
                aos[i].alpha = alpha[i]
        
        NCONS = pts.shape[0]
        aos = DistanceConstraintAOS.field(shape=NCONS)
        kernel(pts, target_pt, pos,target_pos, alpha, aos)
        return aos
    

    def solve(self, pos, target_pos, delta_t=3e-3, maxiter=10):
        """Public API for solving distance constraints"""
        self.aos.lam.fill(0.0)
        for i in range(maxiter):
            self.solve_one_iter(pos,target_pos, delta_t)


    def solve_one_iter(self, pos, target_pos, delta_t):
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
            target_inv_mass: ti.template(),
            pos: ti.template(),
            target_pos: ti.template(),
            delta_t: ti.f32,
            dpos: ti.template(),
        ):
            for i in range(dpos.shape[0]):
                dpos[i] = ti.Vector([0.0, 0.0, 0.0])

            for i in range(aos.shape[0]):
                target_pt, sim_pt = aos[i].p1, aos[i].p2
                invM1, invM2 = target_inv_mass[target_pt], inv_mass[sim_pt]
                l = (target_pos[target_pt] - pos[sim_pt]).norm()
                if l == 0.0:
                    continue
                constraint = l - aos[i].rest_len
                g = (target_pos[target_pt] - pos[sim_pt]) / l
                alpha_tilde = aos[i].alpha / delta_t / delta_t
                dlam = -(constraint + aos[i].lam * alpha_tilde) / (invM1 + invM2 + alpha_tilde)
                aos[i].lam += dlam

                aos[i].dualr = -(constraint + alpha_tilde * aos[i].lam)
                
                if invM2 != 0.0:
                    dpos[sim_pt] -= invM2 * dlam * g


        @ti.kernel
        def solve_distance_constraints_xpbd(
            target_inv_mass:ti.template(),
            inv_mass:ti.template(),
            p1:ti.template(),
            p2:ti.template(),
            rest_len:ti.template(),
            lagrangian:ti.template(),
            dpos:ti.template(),
            pos:ti.template(),
            target_pos:ti.template(),
            alpha:ti.template(),
        ):
            for i in range(NCONS):
                idx0, idx1 = p1[i], p2[i]
                invM0, invM1 = target_inv_mass[idx0], inv_mass[idx1]
                dis = target_pos[idx0] - pos[idx1]
                constraint = dis.norm() - rest_len[i]
                l = -constraint / (invM0 + invM1)
                if l == 0.0:
                    continue
                gradient = dis.normalized()
                alpha_tilde = alpha[i] / delta_t / delta_t
                delta_lagrangian = -(constraint + lagrangian[i] * alpha_tilde) / (invM0 + invM1 + alpha_tilde)
                lagrangian[i] += delta_lagrangian
                
                if invM1 != 0.0:
                    dpos[idx1] -= invM1 * delta_lagrangian * gradient

        NCONS = self.aos.shape[0]
        dpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])
        # _solve_kernel(self.aos, self.inv_mass, self.target_inv_mass, pos, target_pos, delta_t, dpos)
        solve_distance_constraints_xpbd(self.target_inv_mass, self.inv_mass, self.aos.p1, self.aos.p2, self.aos.rest_len, self.aos.lam, dpos, pos, target_pos, self.aos.alpha)
        _update_pos_kernel(self.inv_mass, dpos, self.omega, pos)




@ti.dataclass
class PinToTargetAos:
    p1:ti.i32
    p2:ti.i32

@ti.data_oriented
class PinToTarget():
    """Pin to target geometry"""
    def __init__(self,  pts, target_pt, pos, target_pos):
        NCONS = pts.shape[0]

        self.aos =  self.init_constraints(pts, target_pt, pos, target_pos)
        self.pts = pts
        self.target_pt = target_pt

        self.rest_target_pos = ti.Vector.field(3, dtype=ti.f32, shape=target_pos.shape[0])
        self.rest_target_pos.copy_from(target_pos)
        self.restpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])
        self.restpos.copy_from(pos)

        self.dpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])

    def set_p1(self, target_pt):
        self.aos.p1.from_numpy(target_pt)

    def set_p2(self, pts):
        self.aos.p1.from_numpy(pts)
 
    def init_constraints(self, pts, target_pt, pos, target_pos):
        @ti.kernel
        def kernel(
            pts:ti.template(),
            target_pt:ti.template(),
        ):
            for i in range(pts.shape[0]):
                aos[i].p1, aos[i].p2 = target_pt[i], pts[i]
        
        NCONS = pts.shape[0]
        aos = PinToTargetAos.field(shape=NCONS)
        kernel(pts, target_pt)
        return aos
    
    def solve(self, pos, target_pos):
        """Public API for solving distance constraints"""
        @ti.kernel
        def _kernel(
            aos: ti.template(),
            pos: ti.template(),
            target_pos: ti.template(),
            restpos: ti.template(),
            rest_target_pos: ti.template(),
            dpos: ti.template(),
        ):
            for i in range(aos.shape[0]):
                dpos[aos[i].p2] = target_pos[aos[i].p1] - rest_target_pos[aos[i].p1]
                pos[aos[i].p2] = restpos[aos[i].p2] + dpos[aos[i].p2]
        
        _kernel(self.aos, pos, target_pos, self.restpos, self.rest_target_pos,  self.dpos)
        