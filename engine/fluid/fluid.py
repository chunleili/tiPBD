import taichi as ti
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
        meta.boundary = meta.materials[0]["boundary"]
        meta.gravity = meta.materials[0]["gravity"]
        meta.cell_size = meta.kernel_radius
        meta.num_grid = tuple([math.ceil((meta.boundary[1][d]-meta.boundary[0][d]) / meta.cell_size) for d in range(meta.dim)])
        meta.max_num_particles_per_cell = 60
        meta.max_num_neighbors = 60

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

        self.grid_num_particles = ti.field(int, shape = (meta.num_grid))
        self.grid2particles = ti.field(int, (meta.num_grid + (meta.max_num_particles_per_cell,)))


    def init(self):
        self.pos.from_numpy(self.pos_read)
        self.prev_pos.from_numpy(self.pos_read)

    def substep(self):
        explicit_euler(self.pos, self.prev_pos, self.vel, meta.dt)
        prepare_neighbor_search(self.pos, self.grid_num_particles, self.grid2particles)
        for _ in range(meta.max_iter):
            iteration()
        epilogue()


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
        if p[d] < meta.boundary[0][d]:
            p[d] = meta.boundary[0][d] + meta.padding * ti.random()
        elif p[d] > meta.boundary[1][d]:
            p[d] = meta.boundary[1][d] - meta.padding * ti.random()
    return p


@ti.kernel
def prepare_neighbor_search(pos: ti.template(), grid_num_particles: ti.template(), particle_neighbors: ti.template()):
    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1    


@ti.kernel
def prologue(pos: ti.template(), velocities: ti.template(), old_positions: ti.template(), grid_num_particles: ti.template(), grid2particles: ti.template(), particle_neighbors: ti.template(), particle_num_neighbors: ti.template()):
    # save old positions
    for i in pos:
        old_positions[i] = pos[i]
    # apply gravity within parm.boundary
    for i in pos:
        g = ti.Vector([0.0, -9.8])
        pos, vel = pos[i], velocities[i]
        vel += g * meta.dt
        pos += vel * meta.dt
        pos[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in pos:
        cell = get_cell(pos[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in pos:
        pos_i = pos[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < meta.max_num_neighbors and p_j != p_i and (
                            pos_i - pos[p_j]).norm() < meta.neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

def iteration():
    pass
def epilogue():
    pass

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
