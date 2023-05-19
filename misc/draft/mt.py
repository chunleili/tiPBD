import taichi as ti
import meshtaichi_patcher as pathcher

ti.init(arch=ti.cuda)

mesh = pathcher.load_mesh(
    "model/toy/toy.node",
    relations=[
        "CV",
        "CE",
        "CF",
        "VC",
        "VE",
        "VF",
        "EV",
        "EF",
        "FE",
    ],
)

mesh.verts.place(
    {
        "pos": ti.math.vec3,
    }
)

mesh.verts.pos.from_numpy(mesh.get_position_as_numpy())


@ti.kernel
def test():
    for c in mesh.cells:
        if c.id == 0:
            p0 = c.verts[0]
            p1 = c.verts[1]
            p2 = c.verts[2]
            p3 = c.verts[3]
            pos0 = p0.pos
            pos0 = ti.Vector([1.0, 2.0, 1.0])  # 这里是不能正确更新的！！！
            p1.pos = ti.Vector([3.0, 2.0, 1.0])


@ti.kernel
def test2():
    for c in mesh.cells:
        if c.id == 0:
            print(c.verts[0].pos)  # 注意！！这个没有被正确更新
            print(c.verts[1].pos)  # 而这个则是对的！


test()
test2()
