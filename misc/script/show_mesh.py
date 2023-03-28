import taichi as ti
import meshtaichi_patcher as Patcher

ti.init()


# 读入四面体网格
def init_tet_mesh(model_name):
    #这是用来排列indices的taichi kernel函数
    @ti.kernel
    def init_tet_indices(mesh: ti.template(), indices: ti.template()):
        for c in mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    #基本与上面一样，只是多了一个CV关系，表示通过一个cell可以找到它的四个顶点
    theMesh = Patcher.load_mesh(model_name, relations=["CV"])
    theMesh.verts.place({'x' : ti.math.vec3})
    theMesh.verts.x.from_numpy(theMesh.get_position_as_numpy())
    #这里是四面体，所以每个cell有四个面，每个面有三个顶点，所以indices的长度是len(theMesh.cells) * 4 * 3
    display_indices = ti.field(ti.u32, shape = len(theMesh.cells) * 4 * 3)
    init_tet_indices(theMesh, display_indices)
    return theMesh, display_indices

#这里我们读入了armadillo的四面体模型。这个模型是通过tetgen生成的，我们只需要给出node文件就可以了，它会自动找到ele和face文件。tetgen可以转化ply为node格式，可以在这里下载：http://wias-berlin.de/software/index.jsp?id=TetGen&lang=1
model_name = "model/tetgen/bunny.1.node"
model_name = "model/mygen/bunny2000.node"
# armadillo, armadillo_indices = init_tet_mesh(model_name)
armadillo, armadillo_indices = init_tet_mesh(model_name)

def directly_import_surf():
    face_file_name = "model/mygen/bunny2000.face"
    with open(face_file_name, 'r') as f:
        #skip comments
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        with open(face_file_name, 'r') as f:
            lines = f.readlines()
            NF = int(lines[0].split()[0])
            face_indices = np.zeros((NF, 3), dtype=np.int32)
            for i in range(NF):
                face_indices[i] = np.array(lines[i + 1].split()[1:-1],
                                        dtype=np.int32)

window = ti.ui.Window("taichimesh", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.up(0, 1, 0)
camera.fov(75)
camera.position(4.5,4.5,0.6)
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
    
    # 渲染bunny和armadillo!!
    # scene.mesh(armadillo.verts.x, armadillo_indices, color = (0.5,0.5,0.5), show_wireframe=True)
    scene.mesh(armadillo.verts.x, armadillo_indices, color = (0.5,0.5,0.5))


    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.ambient_light((0.5,0.5,0.5))

    canvas.scene(scene)

    window.show()