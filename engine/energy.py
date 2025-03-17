import taichi as ti
import numpy as np
from pathlib import Path


def compute_energy(inv_mass, pos, predict_pos, tet_indices, B, alpha, dt, is_fixed, fixed_stiffness, fixed_pos):
    pe = compute_ARAP_potential_energy_kernel(pos, tet_indices, B, alpha)
    fe = compute_fixed_energy_kernel(is_fixed, fixed_pos, pos, fixed_stiffness)
    it = compute_inertial_kernel(inv_mass, pos, predict_pos,dt)
    total = pe  + it + fe
    return total

def compute_energy_cuda(extlib,pos, tet_indices, B, rest_volume, mu):
    E = np.array([0.0],dtype=np.float32)
    extlib.compute_energy(pos, pos.shape[0], tet_indices, tet_indices.shape[0], B, rest_volume, mu, E)
    print(E[0])
    # total = pe  + it
    return E[0]


@ti.kernel
def compute_fixed_energy_kernel(is_fixed: ti.template(),
                                fixed_pos: ti.template(),
                                pos: ti.template(),
                                fixed_stiffness: ti.f32) -> ti.f32:
    e = 0.0
    for v in range(pos.shape[0]):
        if is_fixed[v]:
            e += fixed_stiffness / 2 * (pos[v] - fixed_pos[v]).norm_sqr()
    return e


@ti.kernel
def compute_ARAP_potential_energy_kernel(
    pos: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    alpha: ti.template(),
) -> ti.f32:
    pe = 0.0
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id]
        D_s = ti.Matrix.cols([b-a, c-a, d-a])
        F = D_s @ B[i]

        R,_ = ti.polar_decompose(F)

        e = 0.0
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                e += (F[j, k] - R[j, k]) ** 2

        e*=1.0/alpha[i]
        pe += e
    return pe


@ti.kernel
def compute_inertial_kernel(
    inv_mass: ti.template(),
    pos: ti.template(),
    predict_pos: ti.template(),
    dt: ti.f32,
) -> ti.f32:
    it = 0.0
    for i in pos:
        if inv_mass[i]!=0.0:
            it += 1.0/inv_mass[i] * (pos[i] - predict_pos[i]).norm_sqr()
    return it * 0.5 / (dt*dt)