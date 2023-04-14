import taichi as ti
class GGUI():
    def __init__(self) -> None:
        from engine.metadata import meta
        self.window = ti.ui.Window("pbd", (1024, 1024),vsync=False)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()

        self.cam_pos = [-.71, 1.78, 1.77]
        self.cam_lookat = [-.159, -1.1578, 1.213]
        self.camera.position(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        self.camera.lookat(self.cam_lookat[0], self.cam_lookat[1], self.cam_lookat[2])
        self.camera.fov(55) 
        self.gui = self.window.get_gui()
        self.show_widget = True
        self.show_wireframe = True
        self.show_particles = meta.get_common("show_particles", default=True)
        self.show_mesh = meta.get_common("show_mesh", default=False)
        self.par_radius = 0.01
        self.uniform_color = (0.1229,0.2254,0.7207)
        self.par_color = None
        self.show_auxiliary_meshes = True
        if self.show_auxiliary_meshes:
            self.ground, self.coord, self.ground_indices, self.coord_indices = read_auxiliary_meshes()
    
    def update(self, pos_show, indices_show=None):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        self.scene.ambient_light((0.5, 0.5, 0.5))

        if self.show_widget:
            with self.gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
                self.gui.text("camera.curr_position: " + str(self.camera.curr_position))
                self.gui.text("camera.curr_lookat: " + str(self.camera.curr_lookat))
                reset_camera = self.gui.button("Reset Camera")
                if reset_camera:
                    self.camera.position(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
                    self.camera.lookat(self.cam_lookat[0], self.cam_lookat[1], self.cam_lookat[2])
                    self.camera.fov(55) 
        if self.show_mesh:
            self.scene.mesh(pos_show, indices=indices_show, color=self.uniform_color, show_wireframe=self.show_wireframe)
        if self.show_particles:
            self.scene.particles(pos_show, radius=self.par_radius, color=self.uniform_color, per_vertex_color=self.par_color)
        if self.show_auxiliary_meshes:
            self.scene.mesh(self.ground, indices=self.ground_indices, color=(0.5,0.5,0.5), show_wireframe=self.show_wireframe)
            self.scene.mesh(self.coord, indices=self.coord_indices, color=(0.5,0.5,0.5), show_wireframe=self.show_wireframe)

        self.canvas.scene(self.scene)
        self.window.show()


def read_auxiliary_meshes():
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


def visualize(par_pos=None, par_radius=0.01, mesh_pos=None, mesh_indices=None, ti_init=False, background_color=(1,1,1), show_widget=False, par_color=(0.1229,0.2254,0.7207)):
    import numpy as np
    if ti_init:
        ti.init()
    if isinstance(par_pos, np.ndarray):
        par_pos_ti = ti.Vector.field(3, dtype=ti.f32, shape=par_pos.shape[0])
        par_pos_ti.from_numpy(par_pos)
    else:
        par_pos_ti = par_pos

    window = ti.ui.Window("visualizer", (1024, 1024), vsync=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    camera.position(-4.1016811, -1.05783201, 6.2282803)
    camera.lookat(-3.50212255, -0.9375709, 5.43703646)
    camera.fov(55) 
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
            scene.particles(par_pos_ti, radius=par_radius, color=par_color)
        if mesh_pos and mesh_indices is not None:
            scene.mesh(mesh_pos, indices=mesh_indices, color=(0.1229,0.2254,0.7207))

        canvas.scene(scene)
        window.show()



def vis_sdf(grid):
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
        # max = 0.0
        # for i in range(grid.shape[0]):
        #     for j in range(grid.shape[1]):
        #         for k in range(grid.shape[2]):
        #             if grid[i,j,k] > max:
        #                 max = grid[i,j,k]

        for i,j,k in grid:
            if (grid[i,j,k] < threshold):
                par_indx = i * grid.shape[1] * grid.shape[2] + j * grid.shape[2] + k
                particles[par_indx] = ti.Vector([i,j,k]) / grid.shape[0]
                # par_color[par_indx] = grid[i,j,k] / max
    
    occ()

    window = ti.ui.Window("visualizer", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    cam_pos = [-.71, 1.78, 1.77]
    cam_lookat = [-.159, -1.1578, 1.213]

    camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
    camera.lookat(cam_lookat[0], cam_lookat[1], cam_lookat[2])
    camera.fov(55) 
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
                    camera.fov(55) 
        
        scene.particles(particles, radius=0.001, per_vertex_color=par_color)

        canvas.scene(scene)
        window.show()


def test_vis_grid():
    from p2g import p2g
    from debug_info import debug_info
    from visualize import visualize, vis_3d_grid, vis_grid
    from util import random_fill
    ti.init(arch=ti.cuda)
    shape = (100, 100, 100)
    x = ti.field(dtype=ti.f32, shape=shape)
    random_fill(x, 1)
    pass
    vis_grid(x) 

if __name__ == "__main__":
    test_vis_grid()