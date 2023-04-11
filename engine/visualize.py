def visualize(par_pos=None, par_radius=0.01, mesh_pos=None, mesh_indices=None, ti_init=False, background_color=(1,1,1), show_widget=False):
    import taichi as ti
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
            scene.particles(par_pos_ti, radius=par_radius, color=(0.1229,0.2254,0.7207))
        if mesh_pos and mesh_indices is not None:
            scene.mesh(mesh_pos, indices=mesh_indices, color=(0.1229,0.2254,0.7207))

        canvas.scene(scene)
        window.show()