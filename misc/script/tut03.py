import taichi as ti
import meshtaichi_patcher as Patcher

ti.init()


def directly_import_surf():
    import numpy as np
    import os

    pwd = os.getcwd().replace("\\", "/")
    face_file_name = pwd + "/model/mygen/bunny2000_try.face"
    # print("face_file_name: ", face_file_name)
    with open(face_file_name, "r") as f:
        lines = f.readlines()
        NF = int(lines[0].split()[0])
        face_indices = np.zeros((NF, 3), dtype=np.int32)
        for i in range(NF):
            face_indices[i] = np.array(lines[i + 1].split()[1:-1], dtype=np.int32)
    return face_indices.flatten()


armadillo_surf_indices = directly_import_surf()


# 读入四面体网格
def init_tet_mesh(model_name):
    # 基本与上面一样，只是多了一个CV关系，表示通过一个cell可以找到它的四个顶点
    theMesh = Patcher.load_mesh(model_name, relations=["CV"])
    theMesh.verts.place({"x": ti.math.vec3})
    theMesh.verts.x.from_numpy(theMesh.get_position_as_numpy())
    display_indices = ti.field(ti.u32, shape=len(armadillo_surf_indices))
    display_indices.from_numpy(armadillo_surf_indices)
    return theMesh, display_indices


model_name = "model/mygen/bunny2000_try.node"
armadillo, armadillo_indices = init_tet_mesh(model_name)
armadillo_indices.to_numpy()

window = ti.ui.Window("taichimesh", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.up(0, 1, 0)
camera.fov(75)
camera.position(4.5, 4.5, 0.6)
camera.lookat(3.8, 3.8, 0.5)
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
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.mesh(armadillo.verts.x, armadillo_indices, color=(0.5, 0.5, 0.5))

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.show()
