import math
import numpy as np
import taichi as ti
from engine.metadata import meta
from engine.mesh_io import read_particles

dim = 3

# vertices_np,_,_ = read_tetgen(parm.geo_noext)
vertices_np = read_particles(meta.materials[0]["geometry_file"])
vertices_np = vertices_np.astype(np.float32)
num_particles = vertices_np.shape[0]

class Parm:
    pass
parm = Parm()

def set_parm():
    parm.dim = dim
    parm.num_particles = num_particles
    parm.particle_radius = meta.materials[0]["particle_radius"]
    parm.kernel_radius = 4.0 * parm.particle_radius 
    parm.cell_size = 4 * parm.particle_radius
    parm.world_bound = meta.materials[0]["world_bound"]
    parm.num_grid = tuple([math.ceil((parm.world_bound[d][1]-parm.world_bound[d][0]) / parm.cell_size) for d in range(dim)])
    parm.dt = meta.common["dt"]
    parm.num_substeps = meta.common["num_substeps"]

    parm.max_iter = meta.materials[0]["max_iter"]
    parm.gravity = ti.Vector(meta.materials[0]["gravity"])
    parm.epsilon = meta.materials[0]["epsilon"]
    parm.rho0 = meta.materials[0]["rho0"]
    parm.lambda_epsilon = meta.materials[0]["lambda_epsilon"]
    parm.coeff_dq = meta.materials[0]["coeff_dq"]
    parm.coeff_k = meta.materials[0]["coeff_k"]
    parm.neighbor_radius = parm.kernel_radius * 1.05
    parm.poly6_factor = 315.0 / (64.0 * math.pi * math.pow(parm.kernel_radius, 9))
    parm.spiky_grad_factor = -(45.0) / (math.pi * math.pow(parm.kernel_radius, 6))
    parm.max_num_particles_per_cell = 60
    parm.max_num_neighbors = 60

set_parm()

# screen_to_world_ratio = 10.0
positions = ti.Vector.field(dim, float, shape =(num_particles))
old_positions = ti.Vector.field(dim, float, shape = (num_particles))
velocities = ti.Vector.field(dim, float, shape = (num_particles))
grid_num_particles = ti.field(int, shape = (parm.num_grid))
grid2particles = ti.field(int, (parm.num_grid + (parm.max_num_particles_per_cell,)))
particle_num_neighbors = ti.field(int, shape = (num_particles))
particle_neighbors = ti.field(int, shape = ((num_particles,) + (parm.max_num_neighbors,)))
lambdas = ti.field(float, shape = (num_particles))
position_deltas = ti.Vector.field(dim, float, shape=(num_particles))

positions.from_numpy(vertices_np)
old_positions.from_numpy(vertices_np)
debug_info(positions, 'positions')

@ti.func
def poly6_value(r, h):
    res = 0.0
    r2 = r * r
    h2 = h * h
    if r2 <= h2:
        res = math.pow(h2 - r2, 3) * parm.poly6_factor
    return res

@ti.func
def spiky_gradient(r, h):
    res = ti.Vector([0.0, 0.0, 0])
    r2 = r.norm_sqr()
    h2 = h * h
    if r2 <= h2:
        rl = math.sqrt(r2)
        hr = h - rl
        hr2 = hr * hr
        res = parm.spiky_grad_factor * hr2 * r * (1.0 / rl)
    return res


@ti.func
def compute_scorr(pos_ji):
    x = poly6_value(pos_ji.norm(), parm.kernel_radius) / poly6_value(parm.coeff_dq * parm.kernel_radius, parm.kernel_radius)
    x = x * x
    x = x * x
    return (-parm.coeff_k) * x


@ti.func
def get_cell(pos):
    return int(pos * parm.cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < parm.num_grid[0] and 0 <= c[1] and c[
        1] < parm.num_grid[1]


@ti.func
def confine_position_to_boundary(p):
    padding =  4.5 * parm.particle_radius / parm.screen_to_world_ratio
    bmin = padding
    bmax = ti.Vector([parm.boundary[0], parm.boundary[1]
                      ]) - padding
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + parm.epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - parm.epsilon * ti.random()
    return p


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        pos, vel = positions[i], velocities[i]
        vel += parm.gravity * parm.dt
        pos += vel * parm.dt
        positions[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2),(-1,2)))): 
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < parm.max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < parm.neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def iteration():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, parm.kernel_radius)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), parm.kernel_radius)

        # Eq(1)
        density_constraint = (parm.mass * density_constraint / parm.rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                parm.lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, parm.kernel_radius)

        pos_delta_i /= parm.rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / parm.dt
    # no vorticity/xsph because we cannot do cross product in 2D...


def substep():
    prologue()
    for _ in range(parm.max_iter):
        iteration()
    epilogue()

window = ti.ui.Window("pbf", (1024, 1024),vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1.1, 0.0, -1.23)

def render_ggui():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    # print(camera.curr_position)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    # scale_world()
    scene.particles(positions, color = (69/255, 177/255, 232/255), radius = 0.01)
    canvas.scene(scene)
    window.show()

def main():
    print(f'world_bound={parm.world_bound} grid={parm.num_grid} cell_size={parm.cell_size}')
    while window.running:
        for _ in range(parm.num_substeps):
            substep()
        render_ggui()

if __name__ == '__main__':
    main()
