import taichi as ti
import meshtaichi_patcher as patcher
import numpy as np
import taichi.math as tm
from engine.metadata import meta
@ti.data_oriented
class Mesh:
    def __init__(self, model_name, direct_import_faces=True):
        node_file = model_name + ".node"
        self.mesh = patcher.load_mesh(node_file, relations=["CV","CE","CF","VC","VE","VF","EV","EF","FE",])

        self.mesh.verts.place({ 
                                'vel' : ti.math.vec3,
                                'prevPos' : ti.math.vec3,
                                'predictPos' : ti.math.vec3,
                                'invMass' : ti.f32})
        self.mesh.verts.place({ 'pos' : ti.math.vec3},needs_grad=True)
        self.mesh.cells.place({'inv_vol' : ti.f32,
                               'B': ti.math.mat3,
                               'F': ti.math.mat3,
                               'lagrangian': ti.f32,
                               'dLambda': ti.f32,
                               'grad0': ti.math.vec3,
                               'grad1': ti.math.vec3,
                               'grad2': ti.math.vec3,
                               'grad3': ti.math.vec3,
                               'alpha': ti.f32,
                               'fem_constraint': ti.f32,})
        #注意！这里的grad0,1,2,3是针对每个tet的四个顶点的。但是我们把他定义在cell上，而不是vert上。
        #这是因为meshtaichi中vert是唯一的（和几何点是一一对应的）。
        #也就是说多个cell共享同一个顶点时，这个顶点上的数据可能会被覆盖掉。
        #所以这里我们需要为每个tet单独存储grad0,1,2,3。

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())

        self.potential_energy = ti.field(float, (), needs_grad=True)
        self.inertial_energy = ti.field(float, (), needs_grad=True)
        self.total_energy = ti.field(float, (), needs_grad=True)

        # self.lame_lambda = meta.config.get_solids()["lame_lambda"]
        # self.inv_lame_lambda = 1.0 / self.lame_lambda

        self.init_physics()

        # 设置显示三角面的indices
        # 自己计算surf_show
        if not direct_import_faces:
            self.surf_show = ti.field(int, len(self.mesh.cells) * 4 * 3)
            self.init_tet_indices(self.mesh, self.surf_show)
        # 直接读取surf_show
        else:
            surf_show_np = self.directly_import_faces(model_name + '.face')
            self.surf_show = ti.field(ti.i32, surf_show_np.shape[0] * 3)
            self.surf_show.from_numpy(surf_show_np.reshape(surf_show_np.shape[0] * 3))


    @staticmethod
    @ti.kernel
    def init_tet_indices(mesh: ti.template(), indices: ti.template()):
        for c in mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @staticmethod
    def directly_import_faces(face_file_name):
        with open(face_file_name, 'r') as f:
            lines = f.readlines()
            NF = int(lines[0].split()[0])
            face_indices = np.zeros((NF, 3), dtype=np.int32)
            for i in range(NF):
                face_indices[i] = np.array(lines[i + 1].split()[1:-1],
                                        dtype=np.int32)
        return face_indices

    @ti.kernel
    def init_physics(self):
        for v in self.mesh.verts:
            v.invMass = 1.0

        for c in self.mesh.cells:
            p0, p1, p2, p3= c.verts[0].pos, c.verts[1].pos, c.verts[2].pos, c.verts[3].pos
            Dm = tm.mat3([p1 - p0, p2 - p0, p3 - p0])
            c.B = Dm.inverse().transpose()
            c.inv_vol = 6.0/ abs(Dm.determinant()) 
            c.alpha = meta.inv_h2 * meta.inv_lame_lambda * c.inv_vol