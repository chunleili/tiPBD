import math
import numpy as np
import taichi as ti
from engine.metadata import meta
from engine.mesh_io import read_particles

dim = 3

# vertices_np,_,_ = read_tetgen(parm.geo_noext)
vertices_np = read_particles(meta.root_path + "/" + meta.materials[0]["geometry_file"])
vertices_np = vertices_np.astype(np.float32)
num_particles = vertices_np.shape[0]

positions = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles)
positions.from_numpy(vertices_np)

def substep():
    pass

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
    while window.running:
        for _ in range(10):
            substep()
        render_ggui()

if __name__ == '__main__':
    main()
