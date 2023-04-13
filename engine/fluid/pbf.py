import taichi as ti
import numpy as np
from engine.metadata import meta
import math

class Fluid:
    def __init__(self):
        # set parameters
        meta.dim = 3
        meta.max_iter = meta.materials[0]["max_iter"]
        meta.dt = meta.common["dt"]
        meta.particle_radius = meta.materials[0]["particle_radius"]
        meta.kernel_radius = 4 * meta.particle_radius
        meta.padding = 5 * meta.particle_radius
        meta.neighbor_radius = 4 * meta.particle_radius * 1.05
        meta.boundary = meta.materials[0]["boundary"]
        meta.gravity = meta.materials[0]["gravity"]
        meta.cell_size = meta.kernel_radius
        meta.num_grid = tuple([math.ceil((meta.boundary[1][d]-meta.boundary[0][d]) / meta.cell_size) for d in range(meta.dim)])
        meta.max_num_particles_per_cell = 60
        meta.max_num_neighbors = 60
        meta.mass = 0.8 * meta.particle_radius ** meta.dim
        meta.rho0 = meta.materials[0]["rho0"]
        meta.lambda_epsilon = 1e-7
        # meta.poly6_factor = 315.0 / (64.0 * math.pi * math.pow(meta.kernel_radius, 9))
        # meta.spiky_grad_factor = -(45.0) / (math.pi * math.pow(meta.kernel_radius, 6))
        meta.poly6_factor = 315.0 / 64.0 / math.pi
        meta.spiky_grad_factor = -45.0 / math.pi

        # read particles from geometry file
        from engine.mesh_io import read_particles
        import numpy as np
        self.pos_read = read_particles(meta.root_path + "/" + meta.materials[0]["geometry_file"])
        self.pos_read = self.pos_read.astype(np.float32) # cuda 不支持 float64
        self.num_particles = self.pos_read.shape[0]
    
        self.pos = ti.Vector.field(meta.dim, dtype=ti.f32, shape=self.num_particles)
        self.prev_pos = ti.Vector.field(meta.dim, dtype=ti.f32, shape=self.num_particles)
        self.vel = ti.Vector.field(meta.dim, dtype=ti.f32, shape=self.num_particles)
        self.pos_show = self.pos
        self.lambdas = ti.field(ti.f32, shape=self.num_particles)
        self.dpos = ti.Vector.field(meta.dim, dtype=ti.f32, shape=self.num_particles)

        self.grid_num_particles = ti.field(int, shape = (meta.num_grid))
        self.grid2particles = ti.field(int, (meta.num_grid + (meta.max_num_particles_per_cell,)))
        self.particle_neighbors = ti.field(int, (self.num_particles, meta.max_num_neighbors))
        self.particle_num_neighbors = ti.field(int, shape = (self.num_particles))



    def init(self):
        self.pos.from_numpy(self.pos_read)
        self.prev_pos.from_numpy(self.pos_read)

    def substep(self):
        explicit_euler(self.pos, self.prev_pos, self.vel, meta.dt)
        prepare_neighbor_search(self.pos, self.grid_num_particles, self.particle_neighbors, self.grid2particles, self.particle_num_neighbors)
        # from engine.debug_info import debug_info
        # debug_info(self.particle_num_neighbors)
        for _ in range(meta.max_iter):
            iteration(self.pos, self.particle_num_neighbors, self.particle_neighbors, self.lambdas, self.dpos)
        post_solve(self.pos, self.prev_pos, self.vel, meta.dt)


@ti.kernel
def explicit_euler(pos: ti.template(), prev_pos: ti.template(), vel: ti.template(), dt_: ti.f32):
    # semi-Euler update pos & vel
    for i in ti.grouped(pos):
        prev_pos[i] = pos[i]
        vel[i] += meta.gravity * dt_
        pos[i] += vel[i] * dt_
        pos[i] = collision_response(pos[i])


@ti.func
def collision_response(p):
    for d in ti.static(range(meta.dim)):
        if p[d] < meta.boundary[0][d]+ meta.padding:
            p[d] = meta.boundary[0][d] + meta.padding * ti.random()
        elif p[d] > meta.boundary[1][d]- meta.padding:
            p[d] = meta.boundary[1][d] - meta.padding * ti.random()
    return p


@ti.kernel
def prepare_neighbor_search(pos: ti.template(), grid_num_particles: ti.template(), particle_neighbors: ti.template(), grid2particles: ti.template(), particle_num_neighbors: ti.template()):
    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1    
    
    cell_inv = 1 / meta.cell_size
    # update grid
    for p_i in pos:
        cell = ti.cast(pos[p_i] * cell_inv, ti.i32)
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in pos:
        cell = ti.cast(pos[p_i] * cell_inv, ti.i32)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            c = cell + offs
            if is_in_grid(c):
                for k in range(grid_num_particles[c]):
                    p_j = grid2particles[c, k]
                    if nb_i < meta.max_num_neighbors and p_j != p_i and (pos[p_i] - pos[p_j]).norm() < meta.neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.func
def is_in_grid(cell):
    res = True
    for d in ti.static(range(meta.dim)):
        if cell[d] < 0 or cell[d] >= meta.num_grid[d]:
            res = False
    return res

# @ti.kernel
# def prologue(pos: ti.template(), velocities: ti.template(), old_positions: ti.template(), grid_num_particles: ti.template(), grid2particles: ti.template(), particle_neighbors: ti.template(), particle_num_neighbors: ti.template()):
#     # save old positions
#     for i in pos:
#         old_positions[i] = pos[i]
#     # apply gravity within parm.boundary
#     for i in pos:
#         g = ti.Vector([0.0, -9.8])
#         pos, vel = pos[i], velocities[i]
#         vel += g * meta.dt
#         pos += vel * meta.dt
#         pos[i] = confine_position_to_boundary(pos)

#     # clear neighbor lookup table
#     for I in ti.grouped(grid_num_particles):
#         grid_num_particles[I] = 0
#     for I in ti.grouped(particle_neighbors):
#         particle_neighbors[I] = -1

#     # update grid
#     for p_i in pos:
#         cell = get_cell(pos[p_i])
#         # ti.Vector doesn't seem to support unpacking yet
#         # but we can directly use int Vectors as indices
#         offs = ti.atomic_add(grid_num_particles[cell], 1)
#         grid2particles[cell, offs] = p_i
#     # find particle neighbors
#     for p_i in pos:
#         pos_i = pos[p_i]
#         cell = get_cell(pos_i)
#         nb_i = 0
#         for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
#             cell_to_check = cell + offs
#             if is_in_grid(cell_to_check):
#                 for j in range(grid_num_particles[cell_to_check]):
#                     p_j = grid2particles[cell_to_check, j]
#                     if nb_i < meta.max_num_neighbors and p_j != p_i and (
#                             pos_i - pos[p_j]).norm() < meta.neighbor_radius:
#                         particle_neighbors[p_i, nb_i] = p_j
#                         nb_i += 1
#         particle_num_neighbors[p_i] = nb_i

@ti.kernel
def iteration(pos: ti.template(), particle_num_neighbors : ti.template(), particle_neighbors: ti.template(),lambdas: ti.template(), dpos: ti.template()):
    for p_i in pos:
        pos_i = pos[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - pos[p_j]
            grad_j = spiky_gradient(pos_ji, meta.kernel_radius)
            grad_i += grad_j # FIXME：This is BUG
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), meta.kernel_radius)

        # Eq(1)
        density_constraint = (meta.mass * density_constraint / meta.rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                meta.lambda_epsilon)

    # Eq(12), (14)
    for p_i in pos:
        pos_i = pos[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - pos[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, meta.kernel_radius)

        pos_delta_i /= meta.rho0
        dpos[p_i] = pos_delta_i
    # apply position deltas
    for i in pos:
        pos[i] += dpos[i]

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), meta.kernel_radius) / poly6_value(0.3 * meta.kernel_radius,meta.kernel_radius)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-0.1) * x
        
@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = meta.spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = meta.poly6_factor * x * x * x
    return result

@ti.func
def cubic_kernel( r_norm):
    res = ti.cast(0.0, ti.f32)
    h = meta.kernel_radius
    # value of cubic spline smoothing kernel
    k = 1.0
    if meta.dim == 1:
        k = 4 / 3
    elif meta.dim == 2:
        k = 40 / 7 / np.pi
    elif meta.dim == 3:
        k = 8 / np.pi
    k /= h ** meta.dim
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    return res

@ti.func
def cubic_kernel_derivative( r):
    h = meta.kernel_radius
    # derivative of cubic spline smoothing kernel
    k = 1.0
    if meta.dim == 1:
        k = 4 / 3
    elif meta.dim == 2:
        k = 40 / 7 / np.pi
    elif meta.dim == 3:
        k = 8 / np.pi
    k = 6. * k / h ** meta.dim
    r_norm = r.norm()
    q = r_norm / h
    res = ti.Vector([0.0 for _ in range(meta.dim)])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res


@ti.kernel
def post_solve(pos: ti.template(), vel: ti.template(), prev_pos: ti.template(), dt_: ti.f32):
    for i in pos:
        pos[i] = collision_response(pos[i])
    # update velocities
    for i in pos:
        vel[i] = (pos[i] - prev_pos[i]) / dt_


window = ti.ui.Window("pbf", (1024, 1024),vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1.1, 0.0, -1.23)

def ggui_render_particles(pos_show, radius = 0.01):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    # print(camera.curr_position)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    # scale_world()
    scene.particles(pos_show, color = (69/255, 177/255, 232/255), radius = radius)
    canvas.scene(scene)
    window.show()

def main():
    pbd_solver = Fluid()
    pbd_solver.init()
    while window.running:
        for _ in range(10):
            pbd_solver.substep()
        ggui_render_particles(pbd_solver.pos_show)

if __name__ == '__main__':
    main()
