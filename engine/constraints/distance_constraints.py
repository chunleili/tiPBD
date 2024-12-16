import taichi as ti
import numpy as np
from engine.util import timeit
from time import perf_counter


@ti.dataclass
class DistanceConstraintAos:
    idx: ti.i32
    lam:ti.f32
    p1:ti.i32
    p2:ti.i32
    alpha:ti.f32
    rest_len: ti.f32


@ti.kernel
def _solve_distance_attach_kernel(
    inv_mass2:ti.template(),
    p2:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    target_pos:ti.template(),
    alpha:ti.template(),
    delta_t:ti.f32
):
    h2inv = 1.0 / delta_t / delta_t
    for i in range(p2.shape[0]):
        pid = p2[i]
        invM2 =  inv_mass2[pid]
        dis = target_pos[i] - pos[pid]
        l = dis.norm()
        if l == 0.0:
            continue
        constraint = l - rest_len[i]
        gradient = dis / l
        alpha_tilde = alpha[i] * h2inv
        delta_lagrangian = -(constraint + lagrangian[i] * alpha_tilde) / (invM2 + alpha_tilde)
        lagrangian[i] += delta_lagrangian
        
        if invM2 != 0.0:
            dpos[pid] -= invM2 * delta_lagrangian * gradient

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
def _init_restlen_kernel(
    rest_len:ti.template(),
    p2:ti.template(),
    pos:ti.template(),
    target_pos:ti.template() 
):
    for i in range(rest_len.shape[0]):
        rest_len[i] = (target_pos[i] - pos[p2[i]]).norm()

@ti.data_oriented
class DistanceConstraintsAttach():
    """Attach to target geometry"""
    def __init__(self,  pts, pos, target_pos, compliance=1e-9, omega=0.25):
        NCONS = pts.shape[0]

        self.omega = omega
        self.aos =  self.init_constraints(pts, pos, target_pos, compliance)

        self.inv_mass2 = ti.field(dtype=ti.f32, shape=pos.shape[0])
        self.inv_mass2.fill(1.0)

        self.dpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])

    def set_inv_mass2(self, mass:list[float]):
        im = np.array(mass, dtype=np.float32)
        im = 1.0 / im
        self.inv_mass2.from_numpy(im)

    def set_alpha(self, stiffness:list[float]):
        alpha = np.array(stiffness, dtype=np.float32)
        alpha[:] = 1.0 / alpha[:]
        self.aos.alpha.from_numpy(alpha)

    def set_rest_len(self, rest_len:list[float]):
        r = np.array(rest_len, dtype=np.float32)
        self.aos.rest_len.from_numpy(r)

    def set_p2(self, pts:list[int]):
        _  = np.array(pts, dtype=np.int32)
        self.aos.p2.from_numpy(_)
 
    def init_constraints(self, pts, pos, target_pos, compliance):
        NCONS = pts.shape[0]
        aos = DistanceConstraintAos.field(shape=NCONS)
        aos.idx.from_numpy(np.arange(NCONS, dtype=np.int32))
        aos.p2 = pts
        aos.p1.from_numpy(np.ones(NCONS, dtype=np.int32) * -1)
        aos.alpha.from_numpy(np.ones(NCONS, dtype=np.float32) * compliance)
        _init_restlen_kernel(aos.rest_len, aos.p2, pos, target_pos)
        return aos
    
    @timeit
    def solve_one_iter(self, pos, target_pos, delta_t):
        """
        Solve one iteration of the distance constraint

        Usage: First read target pos every frame.  Before the loop: Reset the aos.lam . During the loop: Call this function before solve other constraints.
        """
        _solve_distance_attach_kernel(self.inv_mass2, self.aos.p2, self.aos.rest_len, self.aos.lam, self.dpos, pos, target_pos, self.aos.alpha, delta_t)
        _update_pos_kernel(self.inv_mass2, self.dpos, self.omega, pos)



@ti.kernel
def pintotarget_kernel(
    pts: ti.template(),
    pos: ti.template(),
    target_pos: ti.template(),
    restpos: ti.template(),
    rest_target_pos: ti.template(),
    maxiter:ti.i32
):
    ti.loop_config(serialize=True)
    for i in range(pts.shape[0]):
        # # restvec: vector from rest position to rest target position
        restvec = rest_target_pos[i] - restpos[pts[i]]
        # # toP: position pos to be after the constraint
        toP = target_pos[i] - restvec
        # # dP: displacement from pos to toP
        dP = toP - pos[pts[i]]
        # # move pos to toP in maxiter steps
        pos[pts[i]] += dP/maxiter
        # pos[pts[i]] = toP

@ti.data_oriented
class PinToTarget():
    """Pin to target geometry"""
    def __init__(self,  pts, pos, target_pos):
        self.pts = pts

        self.rest_target_pos = ti.Vector.field(3, dtype=ti.f32, shape=target_pos.shape[0])
        self.rest_target_pos.copy_from(target_pos)
        self.restpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])
        self.restpos.copy_from(pos)

        self.dpos = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])

    def solve(self, pos, target_pos, maxiter):
        pintotarget_kernel(self.pts, pos, target_pos, self.restpos, self.rest_target_pos,   maxiter)
        ...
        