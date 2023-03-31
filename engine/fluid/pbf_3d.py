# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math

import numpy as np

import taichi as ti

ti.init(arch=ti.cpu)

screen_res = (800, 400)
screen_to_world_ratio = 10.0
cell_size = 2.51
cell_recpr = 1.0 / cell_size

grid_size = (32,16)
boundary = (80.0, 40.0)

dim = 2
num_particles = 1200
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

# old_positions = ti.Vector.field(dim, float)
# positions = ti.Vector.field(dim, float)
# velocities = ti.Vector.field(dim, float)
# grid_num_particles = ti.field(int)
# grid2particles = ti.field(int)
# particle_num_neighbors = ti.field(int)
# particle_neighbors = ti.field(int)
# lambdas = ti.field(float)
# position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
# board_states = ti.Vector.field(2, float)

# ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
# grid_snode = ti.root.dense(ti.ij, grid_size)
# grid_snode.place(grid_num_particles)
# grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
# nb_node = ti.root.dense(ti.i, num_particles)
# nb_node.place(particle_num_neighbors)
# nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
# ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
# ti.root.place(board_states)

positions = ti.Vector.field(dim, float, shape =(num_particles))
old_positions = ti.Vector.field(dim, float, shape = (num_particles))
velocities = ti.Vector.field(dim, float, shape = (num_particles))
grid_num_particles = ti.field(int, shape = (grid_size))
grid2particles = ti.field(int, (grid_size + (max_num_particles_per_cell,)))
particle_num_neighbors = ti.field(int, shape = (num_particles))
particle_neighbors = ti.field(int, shape = ((num_particles,) + (max_num_neighbors,)))
lambdas = ti.field(float, shape = (num_particles))
position_deltas = ti.Vector.field(dim, float, shape=(num_particles))
# board_states = ti.Vector.field(2, float, shape =())


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_,
                                                     h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    particle_radius = 3.0
    particle_radius_in_world = particle_radius / screen_to_world_ratio
    bmin = particle_radius_in_world
    bmax = ti.Vector([boundary[0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
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
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def substep():
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
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
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
                spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
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
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...


def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


@ti.kernel
def init_particles():
    num_particles_x = 60   
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4

# @ti.kernel
# def init_particles():
#     init_pos = ti.Vector([10.0,10.0,10.0]) 
#     cube_size = 20
#     spacing = 1
#     num_per_row = (int) (cube_size // spacing) + 1
#     num_per_floor = num_per_row * num_per_row
#     for i in range(num_particles):
#         floor = i // (num_per_floor) 
#         row = (i % num_per_floor) // num_per_row
#         col = (i % num_per_floor) % num_per_row
#         positions[i] = ti.Vector([col*spacing, floor*spacing, row*spacing]) + init_pos


positions_show = ti.Vector.field(dim, float, shape =(num_particles))

@ti.kernel
def scale_world():
    for I in positions:
        for j in ti.static(range(dim)):
            positions_show[I][j] = positions[I][j] * screen_to_world_ratio / screen_res[j]


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

    scale_world()
    scene.particles(positions_show, color = (0.68, 0.26, 0.19), radius = 0.01)
    canvas.scene(scene)
    window.show()

def main():
    init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    while window.running:
        run_pbf()
        render_ggui()

if __name__ == '__main__':
    main()
