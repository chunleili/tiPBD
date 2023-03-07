import taichi as ti
import meshtaichi_patcher as Patcher

ti.init()

#读入表面网格
def init_surf_mesh(model_name):
    @ti.kernel
    def init_surf_indices(mesh: ti.template(), indices: ti.template()):
        for f in mesh.faces:
            for j in ti.static(range(3)):
                indices[f.id * 3 + j] = f.verts[j].id

    theMesh = Patcher.load_mesh(model_name, relations=["FV"])
    theMesh.verts.place({'x' : ti.math.vec3})
    theMesh.verts.x.from_numpy(theMesh.get_position_as_numpy())
    display_indices = ti.field(ti.i32, shape = len(theMesh.faces) * 3)
    init_surf_indices(theMesh, display_indices)
    return theMesh, display_indices


if __name__ == "__main__":
    surf_mesh, surf_mesh_indices = init_surf_mesh("models/beam.obj")

    window = ti.ui.Window("taichimesh", (1024, 1024))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.up(0, 1, 0)
    camera.position(0.25,0.4,-1)
    camera.lookat(0.25,0.4,0)
    camera.fov(75)

    frame = 0
    paused = ti.field(int, shape=())
    paused[None] = 1
    while window.running:
        # 用下面这段代码，通过提前设置一个paused变量，我们就可以在运行的时候按空格暂停和继续了！
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
        if not paused[None]:
            # substep()
            print(f"frame: {frame}")
            frame += 1
        # 我们可以通过下面的代码来查看相机的位置和lookat，这样我们就能知道怎么调整相机的位置了
        # print("camera.curr_position",camera.curr_position)
        # print("camera.curr_lookat",camera.curr_lookat)

        # movement_speed=0.05表示移动速度，hold_key=ti.ui.RMB表示按住右键可以移动视角
        # wasdqe可以移动相机
        camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # 渲染bunny和armadillo!!
        scene.mesh(surf_mesh.verts.x, surf_mesh_indices, color = (0.5,0.5,0.5))

        scene.particles(surf_mesh.verts.x, radius=1e-2, color = (1,0.5,0.5))# 我们也可以把点渲染出来

        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.ambient_light((0.5,0.5,0.5))

        canvas.scene(scene)

        window.show()