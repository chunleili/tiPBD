import taichi as ti
ti.init(ti.gpu)
from engine.fem.arap import *
from engine.metadata import meta
from engine.debug import debug_info
from engine.metadata import meta
@ti.data_oriented
class Solver:
    def run(self):
        # ---------------------------------------------------------------------------- #
        #                                      gui                                     #
        # ---------------------------------------------------------------------------- #
        #init the window, canvas, scene and camerea
        window = ti.ui.Window("pbd", (1024, 1024),vsync=False)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        #initial camera position
        camera.position(-4.1016811, -1.05783201, 6.2282803)
        camera.lookat(-3.50212255, -0.9375709, 5.43703646)
        camera.fov(55) 

        paused = ti.field(int, shape=())
        paused[None] = 1
        meta.step=0
        
        if meta.use_multigrid:
            coarse_to_fine()
        while window.running:
            for e in window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.ESCAPE:
                    exit()
                if e.key == ti.ui.SPACE:
                    paused[None] = not paused[None]
                    print("paused:", paused[None])
                if e.key == "f":
                    print("step: ", meta.step)
                    meta.step+=1
                    substep()
                    # debug_info(mesh.mesh.verts.pos)
                    # debug_info(mesh.mesh.cells.lagrangian)
                    print("step once")

            #do the simulation in each step
            if not paused[None]:
                for _ in range(meta.num_substeps):
                    substep()
                if meta.use_multigrid:
                    coarse_to_fine()

            #set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            # print("camera pos: ", camera.curr_position)
            # print("camera lookat: ", camera.curr_lookat)

            #set the light
            scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
            scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
            scene.ambient_light((0.5, 0.5, 0.5))

            #draw
            if meta.show_coarse:
                scene.mesh(mesh.mesh.verts.pos, indices=mesh.surf_show, color=(0.1229,0.2254,0.7207),show_wireframe=True)
            if meta.show_fine:
                # scene.mesh(fine_pos_ti, indices=fine_tri_idx_ti, color=(1.0,0,0),show_wireframe=True)
                scene.mesh(fine_mesh.mesh.verts.pos, indices=fine_mesh.surf_show, color=(1.0,0,0),show_wireframe=True)

            #show the frame
            canvas.scene(scene)
            window.show()
            # ti.profiler.print_kernel_profiler_info()