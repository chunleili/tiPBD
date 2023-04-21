import taichi as ti
import numpy as np
import trimesh

ti.init(debug=True)

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
    
# pos4 = ti.Vector.field(4, dtype=ti.f32, shape=particles.shape[0])
screen_pos = ti.Vector.field(2, dtype=ti.f32, shape=particles.shape[0])
is_in_rect = ti.field(dtype=ti.i32, shape=particles.shape[0])
per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=particles.shape[0])

per_vertex_color.fill([0.1229,0.2254,0.7207])


mat4x4 = [[0] * 4 for _ in range(4)]
proj_ti = ti.Matrix(mat4x4, dt=ti.f32)
view_ti = ti.Matrix(mat4x4, dt=ti.f32)


def mat4x4_np2ti(mat_ti, mat_np):
    for i in range(4):
        for j in range(4):
            mat_ti[i,j] = mat_np[i,j]


@ti.kernel
def world_to_screen(world_pos: ti.template(), proj: ti.template(), view: ti.template(), screen_pos: ti.template(),screen_w:ti.i32, screen_h:ti.i32):
    for i in world_pos:
        pos_homo = ti.Vector([world_pos[i][0], world_pos[i][1], world_pos[i][2], 1.0])
        ndc =  proj @ view @ pos_homo
        ndc /= ndc[3]

        # screen_pos[i][0] = (ndc[0] + 1.0) * screen_w / 2.0
        # screen_pos[i][1] = (1.0 - ndc[1]) * screen_h / 2.0
        
        # print("world:", world_pos[i])
        # print("screen:", screen_pos[i])



@ti.kernel
def judge_point_in_rect(screen_pos: ti.template(), start: ti.template(), end: ti.template()):
    for i in screen_pos:
        if screen_pos[i][0] > start[0] and screen_pos[i][0] < end[0] and screen_pos[i][1] > start[1] and screen_pos[i][1] < end[1]:
            is_in_rect[i] = True
            per_vertex_color[i] = [1,0,0]
            # print(i)

one_point = ti.Vector.field(3, float, (1))

one_point[0] = [1,1,1]
# one_point[0] = [0,0,0]
# one_point[0] = particle_pos[0]

def visualize(particle_pos):
    window = ti.ui.Window("visualizer", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    screen_w, screen_h = window.get_window_shape()

    # camera.position(-4.1016811, -1.05783201, 6.2282803)
    # camera.lookat(-3.50212255, -0.9375709, 5.43703646)
    camera.position(0,0,1)
    camera.lookat(0,0,0)
    camera.fov(45) 
    canvas.set_background_color((1,1,1))
    
    start = (-1e5,-1e5)
    end   = (1e5,1e5)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))
        # scene.particles(particle_pos, radius=0.01, color=(0.1229,0.2254,0.7207)
        scene.particles(particle_pos, radius=0.01, per_vertex_color=per_vertex_color)
        

        if window.is_pressed(ti.ui.LMB):
            per_vertex_color.fill([0.1229,0.2254,0.7207])
            start = window.get_cursor_pos()
            if window.get_event(ti.ui.RELEASE):
                end = window.get_cursor_pos()
                print("rect start:",start,"\nrect end:", end, "\n")

            rect(start[0], start[1], end[0], end[1])
            canvas.lines(vertices=verts, color=(1,0,0), width=0.005)

            proj = camera.get_projection_matrix(screen_w/screen_h)
            view = camera.get_view_matrix()
            mat4x4_np2ti(proj_ti, proj)
            mat4x4_np2ti(view_ti, view)

            world_to_screen(one_point, proj_ti, view_ti, screen_pos, screen_w, screen_h)
            judge_point_in_rect(screen_pos, start, end)
            pass

        canvas.scene(scene)
        window.show()

visualize(particles_ti)