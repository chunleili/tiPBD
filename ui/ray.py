import taichi as ti
import numpy as np
import trimesh

ti.init(debug=True)

mesh = trimesh.load("data/model/bunny.obj")
particles = mesh.vertices
particles_ti = ti.Vector.field(3, dtype=ti.f32, shape=particles.shape[0])
particles_ti.from_numpy(particles)

def screen_ray(p, view, proj):
    # p is [0,1]x[0,1], first scale it to [-1,1] and get clip space
    pxy = (p - 0.5)*2
    x = pxy[0]
    y = pxy[1]

    inv = np.linalg.inv(proj.transpose() @ view.transpose())
    near = np.array([x,y,-1,1],float)
    far = np.array([x,y,1,1],float)
    near_res = inv @ near
    far_res = inv @ far
    near_res /= near_res[3]
    far_res /= far_res[3]
    dir = np.array([far_res[0]-near_res[0], far_res[1]-near_res[1], far_res[2]-near_res[2]], float)
    dir = dir / np.linalg.norm(dir)
    return dir

ray_show = ti.Vector.field(3, float, 100)
def sample_ray(ray_origin, ray_dir, ray_show):
    for i in range(ray_show.shape[0]):
        ray_show[i] = ray_origin + ray_dir * i * 0.1


def visualize(particle_pos):
    window = ti.ui.Window("visualizer", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    screen_w, screen_h = window.get_window_shape()

    camera.position(0,0,0)
    camera.lookat(0,0,-1)
    camera.fov(45) 
    canvas.set_background_color((1,1,1))
    
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(particle_pos, radius=0.01, color=(0.1229,0.2254,0.7207))

        if window.is_pressed(ti.ui.LMB):
            start = window.get_cursor_pos()
            proj = camera.get_projection_matrix(screen_w/screen_h)
            view = camera.get_view_matrix()
            ray_dir = screen_ray(np.array([start[0],start[1]],float), view, proj)
            ray_origin = (camera.curr_position).to_numpy()
            sample_ray(ray_origin, ray_dir, ray_show)
        scene.particles(ray_show, color=(1,0,1),radius=0.01)
        canvas.scene(scene)
        window.show()
visualize(particles_ti)