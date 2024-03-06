import math
import numpy as np
import taichi as ti

# TODO: 注意！当前的参数是hard-code进去的，实际上并没使用json，后面会改掉。

ti.init()

class Parm:
    pass


parm = Parm()


screen_res = (800, 400)
screen_to_world_ratio = 10.0
parm.cell_size = 2.51
cell_recpr = 1.0 / parm.cell_size

parm.boundary = (screen_res[0] / screen_to_world_ratio, screen_res[1] / screen_to_world_ratio)


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


parm.grid_size = (round_up(parm.boundary[0], 1), round_up(parm.boundary[1], 1))
# print(parm.grid_size)
parm.grid_size = (32, 16)
parm.boundary = (80.0, 40.0)

dim = 2
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 60
num_particles = num_particles_x * 20
max_num_particles_per_cell = 100
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2

parm.max_num_neighbors = 100
parm.time_delta = 1.0 / 20.0
parm.epsilon = 1e-5
parm.particle_radius = 3.0
parm.particle_radius_in_world = parm.particle_radius / screen_to_world_ratio

# PBF params
parm.h_ = 1.1
parm.mass = 1.0
parm.rho0 = 1.0
parm.lambda_epsilon = 100.0
parm.pbf_num_iters = 5
parm.corr_deltaQ_coeff = 0.3
parm.corrK = 0.001
parm.neighbor_radius = parm.h_ * 1.05

parm.poly6_factor = 315.0 / 64.0 / math.pi
parm.spiky_grad_factor = -45.0 / math.pi


positions = ti.Vector.field(dim, float, shape=(num_particles))
old_positions = ti.Vector.field(dim, float, shape=(num_particles))
velocities = ti.Vector.field(dim, float, shape=(num_particles))
# grid_num_particles = ti.field(int, shape = (parm.grid_size))
# grid2particles = ti.field(int, (parm.grid_size + (max_num_particles_per_cell,)))
particle_num_neighbors = ti.field(int, shape=(num_particles))
particle_neighbors = ti.field(int, shape=((num_particles,) + (parm.max_num_neighbors,)))
lambdas = ti.field(float, shape=(num_particles))
position_deltas = ti.Vector.field(dim, float, shape=(num_particles))
board_states = ti.Vector.field(2, float, shape=())


@ti.data_oriented
class SparseGrid:
    def __init__(self, grid_size):
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.grid_snode = ti.root.bitmasked(ti.ij, grid_size)
        self.grid_snode.place(self.grid_num_particles)
        self.grid_snode.bitmasked(ti.k, max_num_particles_per_cell).place(self.grid2particles)

    @ti.kernel
    def usage(self):
        cnt = 0
        for I in ti.grouped(self.grid_snode):
            if ti.is_active(self.grid_snode, I):
                cnt += 1
        usage = cnt / (parm.grid_size[0] * parm.grid_size[1])
        print("Grid usage: ", usage)

    def deactivate(self):
        # ti.deactivate_all_snodes()
        # self.grid_snode.deactivate_all()
        pass


sp = SparseGrid(grid_size=parm.grid_size)
grid_num_particles = sp.grid_num_particles
grid2particles = sp.grid2particles


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = parm.poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = parm.spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), parm.h_) / poly6_value(parm.corr_deltaQ_coeff * parm.h_, parm.h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-parm.corrK) * x


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < parm.grid_size[0] and 0 <= c[1] and c[1] < parm.grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = parm.particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], parm.boundary[1]]) - parm.particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + parm.epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - parm.epsilon * ti.random()
    return p


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * parm.time_delta
    board_states[None] = b


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within parm.boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        pos, vel = positions[i], velocities[i]
        vel += g * parm.time_delta
        pos += vel * parm.time_delta
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
                    if (
                        nb_i < parm.max_num_neighbors
                        and p_j != p_i
                        and (pos_i - positions[p_j]).norm() < parm.neighbor_radius
                    ):
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
            grad_j = spiky_gradient(pos_ji, parm.h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), parm.h_)

        # Eq(1)
        density_constraint = (parm.mass * density_constraint / parm.rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + parm.lambda_epsilon)
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
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, parm.h_)

        pos_delta_i /= parm.rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to parm.boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / parm.time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...


def run_pbf():
    prologue()
    for _ in range(parm.pbf_num_iters):
        substep()
    epilogue()


def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=parm.particle_radius, color=particle_color)
    gui.rect((0, 0), (board_states[None][0] / parm.boundary[0], 1), radius=1.5, color=boundary_color)
    gui.show()


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = parm.h_ * 0.8
        offs = ti.Vector([(parm.boundary[0] - delta * num_particles_x) * 0.5, parm.boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([parm.boundary[0] - parm.epsilon, -0.0])


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")
    sp.usage()


def main():
    init_particles()
    print(f"parm.boundary={parm.boundary} grid={parm.grid_size} parm.cell_size={parm.cell_size}")
    gui = ti.GUI("PBF2D", screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        move_board()
        run_pbf()
        if gui.frame % 20 == 1:
            print_stats()
        render(gui)


# window = ti.ui.Window("pbf", (1024, 1024),vsync=True)
# canvas = window.get_canvas()
# scene = ti.ui.Scene()
# camera = ti.ui.Camera()
# camera.position(1.1, 0.0, -1.23)

# def render_ggui():
#     camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
#     # print(camera.curr_position)
#     scene.set_camera(camera)
#     scene.ambient_light((0.8, 0.8, 0.8))
#     scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

#     # scale_world()
#     scene.particles(positions, color = (69/255, 177/255, 232/255), radius = 0.01)
#     canvas.scene(scene)
#     window.show()

# def main():
#     print(f'parm.boundary={parm.boundary} grid={parm.grid_size} parm.cell_size={parm.cell_size}')
#     while window.running:
#         for _ in range(10):
#             substep()
#         render_ggui()

if __name__ == "__main__":
    main()
