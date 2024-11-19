import taichi as ti
import taichi.math as tm

# ---------------------------------------------------------------------------- #
#                                    kernels                                   #
# ---------------------------------------------------------------------------- #
   
@ti.kernel
def compute_potential_energy_kernel(
    constraints: ti.template(),
    alpha_tilde: ti.template(),
    delta_t: ti.f32,
)->ti.f32:
    potential_energy = 0.0
    for i in range(constraints.shape[0]):
        inv_alpha =  1.0 / (alpha_tilde[i]*delta_t**2)
        potential_energy += 0.5 * inv_alpha * constraints[i]**2
    return potential_energy

@ti.kernel
def compute_inertial_energy_kernel(
    pos: ti.template(),
    predict_pos: ti.template(),
    inv_mass: ti.template(),
    delta_t: ti.f32,
)->ti.f32:
    inertial_energy = 0.0
    inv_h2 = 1.0 / delta_t**2
    for i in range(pos.shape[0]):
        if inv_mass[i] == 0.0:
            continue
        inertial_energy += 0.5 / inv_mass[i] * (pos[i] - predict_pos[i]).norm_sqr() * inv_h2
    return inertial_energy

@ti.kernel
def calc_dual_kernel(alpha_tilde:ti.template(),
                       lagrangian:ti.template(),
                       constraints:ti.template(),
                       dual_residual:ti.template())->ti.f32:
    dual = 0.0
    for i in range(dual_residual.shape[0]):
        dual_residual[i] = -(constraints[i] + alpha_tilde[i] * lagrangian[i])
        dual += dual_residual[i] * dual_residual[i]
    dual = ti.sqrt(dual)
    return dual



@ti.kernel
def update_pos_kernel(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    omega:ti.f32
):
    for i in range(inv_mass.shape[0]):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]

            
@ti.kernel
def semi_euler_kernel(
    delta_t: ti.f32,
    pos: ti.template(),
    predict_pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    damping_coeff: ti.f32,
    gravity: ti.template(),
    inv_mass: ti.template(),
    force: ti.types.ndarray(dtype=tm.vec3),
):
    for i in pos:
        if inv_mass[i] != 0.0:
            old_pos[i] = pos[i]
            vel[i] += damping_coeff* delta_t * (gravity + force[i])
            pos[i] += delta_t * vel[i]
            predict_pos[i] = pos[i]


@ti.kernel
def update_vel_kernel(delta_t: ti.f32,
                      pos: ti.template(),
                      old_pos: ti.template(),
                      vel: ti.template(),
                      inv_mass: ti.template()):
    for i in pos:
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / delta_t




# ground collision response
@ti.kernel
def ground_collision_kernel(pos: ti.template(), old_pos:ti.template(), ground_pos: ti.f32, inv_mass: ti.template()):
    for i in pos:
        if inv_mass[i] != 0.0:
            if pos[i][1] < ground_pos:
                pos[i] = old_pos[i]
                pos[i][1] = ground_pos

@ti.kernel
def calc_norm_kernel(a:ti.template())->ti.f32:
    sum = 0.0
    for i in range(a.shape[0]):
        sum += a[i] * a[i]
    sum = ti.sqrt(sum)
    return sum
# ---------------------------------------------------------------------------- #
#                                  end kernels                                 #
# ---------------------------------------------------------------------------- #