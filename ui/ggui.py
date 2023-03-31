import taichi as ti
from engine.metadata import meta
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

        if meta.show_coarse:
            self.scene.mesh(pos_show, indices=indices_show, color=(0.1229,0.2254,0.7207),show_wireframe=True)
        # if meta.show_fine:
            # self.scene.mesh(fine_pos_ti, indices=fine_tri_idx_ti, color=(1.0,0,0),show_wireframe=True)
            # self.scene.mesh(fine_mesh.mesh.verts.pos, indices=fine_mesh.surf_show, color=(1.0,0,0),show_wireframe=True)

        self.canvas.scene(self.scene)
        self.window.show()