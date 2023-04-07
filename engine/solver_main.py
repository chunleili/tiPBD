import taichi as ti

def solver_main():
    ggui = GGUI()
    from engine.metadata import meta
    if meta.common['constitutive_model'] == 'arap':
        from engine.fem.arap import ARAP
        pbd_solver = ARAP()
    elif meta.common['constitutive_model'] == 'neohooken':
        from engine.fem.neohooken import NeoHooken
        pbd_solver = NeoHooken()
    else:
        raise NotImplementedError
    
    meta.paused = meta.common["initial_pause"]
    
    # if meta.use_multigrid:
        # coarse_to_fine()
    while ggui.window.running:
        for e in ggui.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                meta.paused = not meta.paused
                print("paused:", meta.paused)
            if e.key == "f":
                print("step: ", meta.step)
                meta.step+=1
                pbd_solver.substep()
                # debug_info(mesh.mesh.verts.pos)
                # debug_info(mesh.mesh.cells.lagrangian)
                print("step once")

        #do the simulation in each step
        if not meta.paused:
            for _ in range(meta.num_substeps):
                pbd_solver.substep()
                # print("pbd_solver.mesh.mesh.verts.pos",pbd_solver.mesh.mesh.verts.pos)
            # if meta.use_multigrid:
                # coarse_to_fine()
            
        ggui.update(pbd_solver.mesh.mesh.verts.pos, pbd_solver.mesh.surf_show)
    
        if meta.args.kernel_profiler:
            ti.profiler.print_kernel_profiler_info()


class GGUI():
    def __init__(self) -> None:
        self.window = ti.ui.Window("pbd", (1024, 1024),vsync=False)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()

        self.camera.position(-4.1016811, -1.05783201, 6.2282803)
        self.camera.lookat(-3.50212255, -0.9375709, 5.43703646)
        self.camera.fov(55) 
    
    def update(self, pos_show, indices_show):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        # print("self.camera pos: ", self.camera.curr_position)
        # print("self.camera lookat: ", self.camera.curr_lookat)

        self.scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        self.scene.ambient_light((0.5, 0.5, 0.5))
        
        from engine.metadata import meta
        if meta.show_coarse:
            self.scene.mesh(pos_show, indices=indices_show, color=(0.1229,0.2254,0.7207),show_wireframe=True)
        # if meta.show_fine:
            # self.scene.mesh(fine_pos_ti, indices=fine_tri_idx_ti, color=(1.0,0,0),show_wireframe=True)
            # self.scene.mesh(fine_mesh.mesh.verts.pos, indices=fine_mesh.surf_show, color=(1.0,0,0),show_wireframe=True)

        self.canvas.scene(self.scene)
        self.window.show()