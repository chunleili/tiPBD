import taichi as ti
import trimesh

ti.init()

mesh = trimesh.load("data/model/bunny.obj")
particles = mesh.vertices
particles_ti = ti.Vector.field(3, dtype=ti.f32, shape=particles.shape[0])
particles_ti.from_numpy(particles)


verts = ti.Vector.field(2, dtype=ti.f32, shape=8)

def rect(x_min, y_min, x_max, y_max):
    verts[0] = [x_min, y_min]
    verts[1] = [x_max, y_min]
    verts[2] = [x_min, y_max]
    verts[3] = [x_max, y_max]
    verts[4] = [x_min, y_min]
    verts[5] = [x_min, y_max]
    verts[6] = [x_max, y_min]
    verts[7] = [x_max, y_max]
    
def visualize(particle_pos):
    window = ti.ui.Window("visualizer", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    camera.position(-4.1016811, -1.05783201, 6.2282803)
    camera.lookat(-3.50212255, -0.9375709, 5.43703646)
    camera.fov(55) 
    canvas.set_background_color((1,1,1))
    
    start = (-1e5,-1e5)
    end   = (1e5,1e5)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(particle_pos, radius=0.01, color=(0.1229,0.2254,0.7207))
        
        if window.is_pressed(ti.ui.LMB):
            start = window.get_cursor_pos()
            if window.get_event(ti.ui.RELEASE):
                end = window.get_cursor_pos()
            rect(start[0], start[1], end[0], end[1])
            canvas.lines(vertices=verts, color=(1,0,0), width=0.005)
            
        canvas.scene(scene)
        window.show()

visualize(particles_ti)