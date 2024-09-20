import taichi as ti
import math

ti.init()

num_particles = 2000
dim = 3
world_scale_factor = 1.0 / 100.0
dt = 1e-2
mass_inv = 1.0

positions = ti.Vector.field(dim, float, num_particles)
velocities = ti.Vector.field(dim, float, num_particles)
pos_show = ti.Vector.field(dim, float, num_particles)
force = ti.Vector.field(dim, float, num_particles)
positions0 = ti.Vector.field(dim, float, num_particles)
radius_vector = ti.Vector.field(dim, float, num_particles)
paused = ti.field(ti.i32, shape=())
q_inv = ti.Matrix.field(n=3, m=3, dtype=float, shape=())


@ti.kernel
def init_cube():
    init_pos = ti.Vector([70.0, 50.0, 0.0])
    cube_size = 20
    spacing = 2
    num_per_row = (int)(cube_size // spacing) + 1
    num_per_floor = num_per_row * num_per_row
    for i in range(num_particles):
        floor = i // (num_per_floor)
        row = (i % num_per_floor) // num_per_row
        col = (i % num_per_floor) % num_per_row
        positions[i] = ti.Vector([col * spacing, floor * spacing, row * spacing]) + init_pos


read_mesh_from_file = True


def init_particles():
    if read_mesh_from_file:
        import os, sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from engine.mesh_io import read_mesh
        from engine.util import meta

        scale = meta.get_materials("scale")
        shift = meta.get_materials("translation")
        pos_np, indices_show_np = read_mesh(meta.get_materials("geometry_file"), scale=scale, shift=shift)
        pos_np /= world_scale_factor  # 因为有个world_scale_factor，所以这里要除以这个因子
        positions.from_numpy(pos_np)
        return indices_show_np.flatten()
    else:
        init_cube()
        rotation(60)  # initially rotate the cube
        return None


@ti.kernel
def shape_matching():
    #  update vel and pos firtly
    gravity = ti.Vector([0.0, -9.8, 0.0])
    for i in range(num_particles):
        positions0[i] = positions[i]
        f = gravity
        velocities[i] += mass_inv * f * dt
        positions[i] += velocities[i] * dt
        if positions[i].y < 0.0:
            positions[i] = positions0[i]
            positions[i].y = 0.0

    # compute the new(matched shape) mass center
    c = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_particles):
        c += positions[i]
    c /= num_particles

    # compute transformation matrix and extract rotation
    A = sum1 = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f32)
    for i in range(num_particles):
        sum1 += (positions[i] - c).outer_product(radius_vector[i])
    A = sum1 @ q_inv[None]

    R, _ = ti.polar_decompose(A)

    # update velocities and positions
    for i in range(num_particles):
        positions[i] = c + R @ radius_vector[i]
        velocities[i] = (positions[i] - positions0[i]) / dt


@ti.kernel
def compute_radius_vector():
    # compute the mass center and radius vector
    center_mass = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_particles):
        center_mass += positions[i]
    center_mass /= num_particles
    for i in range(num_particles):
        radius_vector[i] = positions[i] - center_mass


@ti.kernel
def precompute_q_inv():
    res = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f64)
    for i in range(num_particles):
        res += radius_vector[i].outer_product(radius_vector[i])
    q_inv[None] = res.inverse()


@ti.kernel
def rotation(angle: ti.f32):
    theta = angle / 180.0 * math.pi
    R = ti.Matrix([[ti.cos(theta), -ti.sin(theta), 0.0], [ti.sin(theta), ti.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    for i in range(num_particles):
        positions[i] = R @ positions[i]


# ---------------------------------------------------------------------------- #
#                                    substep                                   #
# ---------------------------------------------------------------------------- #
def substep():
    shape_matching()


# ---------------------------------------------------------------------------- #
#                                  end substep                                 #
# ---------------------------------------------------------------------------- #


@ti.kernel
def world_scale():
    for i in range(num_particles):
        pos_show[i] = positions[i] * world_scale_factor


# init the window, canvas, scene and camerea
window = ti.ui.Window("rigidbody", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
canvas.set_background_color((1, 1, 1))

# initial camera position
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(45)


def main():
    indices_show_np = init_particles()
    if indices_show_np is not None:
        indices_show = ti.field(ti.i32, shape=indices_show_np.shape)
        indices_show.from_numpy(indices_show_np)

    compute_radius_vector()  # store the shape of rigid body
    precompute_q_inv()

    from engine.visualize import draw_bounds, read_auxiliary_meshes

    anchors, line_indices = draw_bounds()
    ground, coord, ground_indices, coord_indices = read_auxiliary_meshes()

    paused[None] = True
    while window.running:
        if window.get_event(ti.ui.PRESS):
            # press space to pause the simulation
            if window.event.key == ti.ui.SPACE:
                paused[None] = not paused[None]

            # proceed once for debug
            if window.event.key == "p":
                substep()

        # do the simulation in each step
        if paused[None] == False:
            for i in range(int(0.05 / dt)):
                substep()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw particles
        world_scale()
        scene.particles(pos_show, radius=0.005, color=(0, 0, 1))
        scene.lines(vertices=anchors, indices=line_indices, color=(1, 0, 0), width=2)
        scene.mesh(ground, indices=ground_indices, color=(0.5, 0.5, 0.5))
        scene.mesh(coord, indices=coord_indices, color=(0.5, 0, 0))
        # if indices_show_np is not None:
        #     scene.mesh(pos_show, indices=indices_show, color=(0, 0, 1))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
