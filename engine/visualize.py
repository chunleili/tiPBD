import taichi as ti
class GGUI():
    def __init__(self) -> None:
        from engine.metadata import meta
        self.vsync = meta.get_common("sync", default=False)
        self.window = ti.ui.Window("pbd", (1024, 1024),vsync=self.vsync)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()

        self.canvas.set_background_color((1,1,1))
        self.cam_pos = meta.get_common("camera_pos", default=[.5, .5 , 2])
        self.cam_lookat = meta.get_common("camera_lookat", default=[.5, .5, 1])
        self.camera.position(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        self.camera.lookat(self.cam_lookat[0], self.cam_lookat[1], self.cam_lookat[2])
        self.camera.fov(45) 
        self.gui = self.window.get_gui()
        meta.show_widget = meta.get_common("show_widget", default=True)
        meta.show_wireframe = meta.get_common("show_wireframe", default=False)
        meta.show_particles = meta.get_common("show_particles", default=True)
        meta.show_mesh = meta.get_common("show_mesh", default=True)
        meta.show_auxiliary_meshes = meta.get_common("show_auxiliary_meshes", default=True)
        meta.show_bounds = meta.get_common("show_bounds", default=True)
        meta.show_sdf = meta.get_common("show_sdf", default=True)
        meta.particle_radius_show = meta.get_common("particle_radius_show", default=0.002)
        self.uniform_color = (0.1229,0.2254,0.7207)
        self.par_color = None
        meta.mesh_uniform_color = meta.get_common("mesh_uniform_color", default=self.uniform_color)
        meta.mesh_uniform_color = tuple(meta.mesh_uniform_color)
        meta.particle_uniform_color = meta.get_common("particle_uniform_color", default=self.uniform_color)
        meta.particle_uniform_color = tuple(meta.particle_uniform_color)

        if meta.show_auxiliary_meshes:
            self.ground, self.coord, self.ground_indices, self.coord_indices = read_auxiliary_meshes()
        
        if meta.show_bounds:
            self.box_anchors, self.box_lines_indices = draw_bounds(x_min=0, y_min=0, z_min=0, x_max=1, y_max=1, z_max=1)

        meta.use_sdf =  meta.get_common("use_sdf")

    
    def update(self, pos_show, indices_show=None):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        self.scene.ambient_light((0.5, 0.5, 0.5))
        
        from engine.metadata import meta

        if meta.show_widget:
            with self.gui.sub_window("Options", 0, 0, 0.3, 0.45) as w:
                self.gui.text("cam pos: " + f"{self.camera.curr_position}")
                self.gui.text("cam lookat: " + f"{self.camera.curr_lookat}")
                reset_camera = self.gui.button("Reset Camera")
                if reset_camera:
                    self.camera.position(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
                    self.camera.lookat(self.cam_lookat[0], self.cam_lookat[1], self.cam_lookat[2])
                    self.camera.fov(45) 
                meta.num_substeps = self.gui.slider_int("num_substeps", meta.num_substeps, 0, 100)
                if hasattr(meta, "max_iter"):
                    meta.max_iter = self.gui.slider_int("max_iter", meta.max_iter, 0, 50)
                if hasattr(meta, "relax_factor"):
                    meta.relax_factor = self.gui.slider_float("relax_factor", meta.relax_factor, 0, 1)
                self.gui.text("step number: " + str(meta.step_num))
                # self.gui.text("current iter: " + str(getattr(meta,"iter", 0))+"/"+str(meta.max_iter))
                if self.gui.button("show particles"):
                    meta.show_particles = not meta.show_particles
                meta.particle_radius_show = self.gui.slider_float("particle radius", meta.particle_radius_show, 0, 0.01)
                meta.particle_uniform_color = self.gui.color_edit_3("particle color", meta.particle_uniform_color)
                meta.mesh_uniform_color = self.gui.color_edit_3("mesh color", meta.mesh_uniform_color)
                if self.gui.button("show mesh"):
                    meta.show_mesh = not meta.show_mesh
                if self.gui.button("show auxiliary meshes"):
                    meta.show_auxiliary_meshes = not meta.show_auxiliary_meshes
                if self.gui.button("show bounds"):
                    meta.show_bounds = not meta.show_bounds
                if self.gui.button("show sdf"):
                    meta.show_sdf = not meta.show_sdf
                if self.gui.button("show wireframe"):
                    meta.show_wireframe = not meta.show_wireframe
                if self.gui.button("pause"):
                    meta.paused = not meta.paused
                self.gui.text("paused: " + str(meta.paused))
                if self.gui.button("selector on/off"):
                    meta.use_selector = not meta.use_selector
                self.gui.text("use selector: " + str(meta.use_selector))
                

        if indices_show is not None and meta.show_mesh:
            self.scene.mesh(pos_show, indices=indices_show, color=meta.mesh_uniform_color, show_wireframe=meta.show_wireframe)
        if meta.show_particles:
            self.scene.particles(pos_show, radius=meta.particle_radius_show, color=meta.particle_uniform_color)
        if meta.show_auxiliary_meshes:
            self.scene.mesh(self.ground, indices=self.ground_indices, color=(0.5,0.5,0.5), show_wireframe=meta.show_wireframe)
            self.scene.mesh(self.coord, indices=self.coord_indices, color=(0.5, 0, 0), show_wireframe=meta.show_wireframe)

        if meta.show_bounds:
            self.scene.lines(self.box_anchors, indices=self.box_lines_indices, color = (0.99, 0.68, 0.28), width = 2.0)

        if meta.use_sdf and meta.show_sdf:
            self.scene.particles(self.sdf_vertices, radius=meta.particle_radius_show, color=self.particle_uniform_color)

        # if meta.get_common("vis_sparse_grid"):
        #     self.scene.lines(self.sparse_grid_anchors, indices=self.sparse_grid_indices, color = (0.99, 0.68, 0.28), width = 2.0)


def read_auxiliary_meshes():
    '''
    读取辅助网格，包括地面和坐标系。
    
    Examples::

        # (before the render loop)
        ground, coord, ground_indices, coord_indices = read_auxiliary_meshes()
        # ...
        # (in the render loop)
        scene.mesh(ground, indices=ground_indices, color=(0.5,0.5,0.5))
        scene.mesh(coord, indices=coord_indices, color=(0.5, 0, 0))
    '''
    from engine.mesh_io import read_mesh
    ground_, ground_indices_ = read_mesh("data/model/ground.obj")
    coord_, coord_indices_ = read_mesh("data/model/coord.obj")
    ground_indices_ = ground_indices_.flatten()
    coord_indices_ = coord_indices_.flatten()
    ground = ti.Vector.field(3, dtype=ti.f32, shape=ground_.shape[0])
    ground.from_numpy(ground_)
    coord = ti.Vector.field(3, dtype=ti.f32, shape=coord_.shape[0])
    coord.from_numpy(coord_)
    ground_indices = ti.field(dtype=ti.i32, shape=ground_indices_.shape[0])
    ground_indices.from_numpy(ground_indices_)
    coord_indices = ti.field(dtype=ti.i32, shape=coord_indices_.shape[0])
    coord_indices.from_numpy(coord_indices_)

    return ground, coord, ground_indices, coord_indices


def visualize(par_pos=None, particle_radius_show=0.01, mesh_pos=None, mesh_indices=None, ti_init=False, background_color=(1,1,1), show_widget=False, par_color=(0.1229,0.2254,0.7207)):
    import numpy as np
    if ti_init:
        ti.init()
    if isinstance(par_pos, np.ndarray):
        par_pos_ti = ti.Vector.field(3, dtype=ti.f32, shape=par_pos.shape[0])
        par_pos_ti.from_numpy(par_pos)
    else:
        par_pos_ti = par_pos

    window = ti.ui.Window("visualizer", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    camera.position(-4.1016811, -1.05783201, 6.2282803)
    camera.lookat(-3.50212255, -0.9375709, 5.43703646)
    camera.fov(45) 
    canvas.set_background_color(background_color)
    gui = window.get_gui()
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        if show_widget:
            with gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
                gui.text("camera.curr_position: " + str(camera.curr_position))
                gui.text("camera.curr_lookat: " + str(camera.curr_lookat))
        
        if par_pos is not None:
            scene.particles(par_pos_ti, radius=particle_radius_show, color=par_color)
        if mesh_pos and mesh_indices is not None:
            scene.mesh(mesh_pos, indices=mesh_indices, color=(0.1229,0.2254,0.7207))

        canvas.scene(scene)
        window.show()



def vis_sdf(grid, provide_render=True):
    import numpy as np
    num_particles = grid.shape[0] * grid.shape[1] * grid.shape[2]
    particles = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
    par_color = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

    if  isinstance(grid, np.ndarray):
        grid_ = ti.field(dtype=ti.f32, shape=grid.shape)
        grid_.from_numpy(grid)
        grid = grid_

    threshold = 0.0
    @ti.kernel
    def occ():
        for i,j,k in grid:
            if (grid[i,j,k] < threshold):
                par_indx = i * grid.shape[1] * grid.shape[2] + j * grid.shape[2] + k
                particles[par_indx] = ti.Vector([i,j,k]) / grid.shape[0]
                # par_color[par_indx] = grid[i,j,k] / max
    occ()

    if provide_render:
        window = ti.ui.Window("visualizer", (1024, 1024), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        cam_pos = [-.71, 1.78, 1.77]
        cam_lookat = [-.159, -1.1578, 1.213]

        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_lookat[0], cam_lookat[1], cam_lookat[2])
        camera.fov(45) 
        canvas.set_background_color((1,1,1))
        gui = window.get_gui()
        show_widget = True
        reset_camera = False
        while window.running:
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
            scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
            scene.ambient_light((0.5, 0.5, 0.5))

            if show_widget:
                with gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
                    gui.text("camera.curr_position: " + str(camera.curr_position))
                    gui.text("camera.curr_lookat: " + str(camera.curr_lookat))
                    reset_camera = gui.button("Reset Camera")
                    if reset_camera:
                        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
                        camera.lookat(cam_lookat[0], cam_lookat[1], cam_lookat[2])
                        camera.fov(45) 
            
            scene.particles(particles, radius=0.001, per_vertex_color=par_color)

            canvas.scene(scene)
            window.show()
    return particles

# FIXME
def vis_sparse_grid(grid, resolution):
    threshold = 0.0
    dx = 1.0 / resolution
    anchors = ti.Vector.field(3, dtype=ti.f32, shape = (resolution,resolution,resolution,8))
    indices = ti.field(dtype=ti.i32, shape = (resolution,resolution,resolution, 24))

    @ti.kernel
    def occ():
        for i,j,k in grid:
            if (grid[i,j,k] < threshold):
                x_min = i * dx
                x_max = (i+1) * dx
                y_min = j * dx
                y_max = (j+1) * dx
                z_min = k * dx
                z_max = (k+1) * dx
                anchors[i,j,k, 0] = ti.Vector([x_min, y_min, z_min])
                anchors[i,j,k, 1] = ti.Vector([x_min, y_max, z_min])
                anchors[i,j,k, 2] = ti.Vector([x_max, y_min, z_min])
                anchors[i,j,k, 3] = ti.Vector([x_max, y_max, z_min])
                anchors[i,j,k, 4] = ti.Vector([x_min, y_min, z_max])
                anchors[i,j,k, 5] = ti.Vector([x_min, y_max, z_max])
                anchors[i,j,k, 6] = ti.Vector([x_max, y_min, z_max])
                anchors[i,j,k, 7] = ti.Vector([x_max, y_max, z_max])

                for l, val in ti.static(enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7])):
                    indices[i,j,k, l] = val
    occ()
    return anchors, indices


def draw_bounds(x_min=0, y_min=0, z_min=0, x_max=1, y_max=1, z_max=1):
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    box_anchors[0] = ti.Vector([x_min, y_min, z_min])
    box_anchors[1] = ti.Vector([x_min, y_max, z_min])
    box_anchors[2] = ti.Vector([x_max, y_min, z_min])
    box_anchors[3] = ti.Vector([x_max, y_max, z_min])
    box_anchors[4] = ti.Vector([x_min, y_min, z_max])
    box_anchors[5] = ti.Vector([x_min, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, y_min, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))
    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val
    return box_anchors, box_lines_indices