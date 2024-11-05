"""
Build the cloth mesh
"""


import taichi as ti
import numpy as np

@ti.data_oriented
class TriMeshCloth:
    def __init__(self, mesh_file) -> None:
        self.mesh_file = mesh_file
        # FIXME setup_num

    def fetch_fields(self, pos, inv_mass, edge, rest_len):
        pos.from_numpy(self.pos)
        inv_mass.from_numpy(self.inv_mass)
        edge.from_numpy(self.edge)
        rest_len.from_numpy(self.rest_len)

    def build(self):
        self.pos, self.edge, self.tri = self.read_tri_cloth_obj(self.mesh_file)
        self.NV = len(self.pos)
        self.NT = len(self.tri)
        self.NE = len(self.edge)
        self.rest_len = np.zeros(self.NE, dtype=np.float32)
        self.inv_mass = np.ones(self.NV, dtype=np.float32)
        self.init_rest_len(self.edge, self.rest_len, self.pos)
        self.fixed_points = [0, self.NV-1]
        self.init_mass(self.inv_mass, self.fixed_points)
        return self.pos, self.edge, self.tri, self.inv_mass, self.rest_len


    @staticmethod
    def read_tri_cloth_obj(path):
        import meshio
        print(f"path is {path}")
        mesh = meshio.read(path)
        tri = mesh.cells_dict["triangle"]
        pos = mesh.points

        num_tri = len(tri)
        edges=[]
        for i in range(num_tri):
            ele = tri[i]
            # [0,1][1,2][2,0]: the counter-clockwise order, we use 
            # [0,2][2,1][1,0]: the clockwise order, also reasonable
            edges.append([min((ele[0]), (ele[1])), max((ele[0]),(ele[1]))])
            edges.append([min((ele[1]), (ele[2])), max((ele[1]),(ele[2]))])
            edges.append([min((ele[0]), (ele[2])), max((ele[0]),(ele[2]))])
        #remove the duplicate edges
        # https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
        import itertools
        edges.sort()
        edges = list(edges for edges,_ in itertools.groupby(edges))
        return pos, np.array(edges), tri
    

    @staticmethod
    @ti.kernel
    def init_rest_len(
        edge:ti.types.ndarray(),
        rest_len:ti.types.ndarray(),
        pos:ti.types.ndarray(dtype=ti.types.vector(3,float)),
    ):
        for i in range(edge.shape[0]):
            idx1, idx2 = edge[i,0], edge[i,1]
            p1, p2 = pos[idx1], pos[idx2]
            rest_len[i] = (p1 - p2).norm()

    @staticmethod
    def init_mass(inv_mass:np.ndarray, fixed_points:list):
        inv_mass[:] = 1.0
        for i in fixed_points:
            inv_mass[i] = 0.0
    





@ti.data_oriented
class QuadMeshCloth():
    def __init__(self, N, setup_num=0) -> None:
        self.N = N
        self.NV = (N + 1)**2
        self.NT = 2 * N**2
        self.NE = 2 * N * (N + 1) + N**2
        self.NCONS = self.NE
        self.setup_num = setup_num # 0: fixed point, 1: strech and no fixed point
    
    def build(self):
        # self.create_fields()
        # init topology
        self.init_tri(self.tri, self.N)
        self.init_edge(self.edge, self.N)

        # init_physical
        self.init_pos(self.pos, self.N)
        self.init_mass(self.inv_mass, self.N, self.NV, self.setup_num )
        self.init_rest_len(self.edge, self.rest_len, self.pos)


    # pass fields from outside, not used
    def pass_fields(self,pos, inv_mass, edge, tri, rest_len):
        self.pos = pos
        self.inv_mass = inv_mass
        self.edge = edge
        self.tri = tri
        self.rest_len = rest_len

    def create_fields(self):
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)
        self.inv_mass = ti.field(dtype=ti.f32, shape=self.NV)
        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.NE)
        self.tri = ti.field(ti.i32, shape=3 * self.NT)
        self.rest_len = ti.field(dtype=ti.f32, shape=self.NE)

    def to_numpy(self):
        self.pos = self.pos.to_numpy()
        self.inv_mass = self.inv_mass.to_numpy()
        self.edge = self.edge.to_numpy()
        self.tri = self.tri.to_numpy().reshape(-1, 3)
        self.rest_len = self.rest_len.to_numpy()

    @staticmethod
    @ti.kernel
    def init_rest_len(
        edge:ti.template(),
        rest_len:ti.template(),
        pos:ti.template(),
    ):
        for i in range(edge.shape[0]):
            idx1, idx2 = edge[i]
            p1, p2 = pos[idx1], pos[idx2]
            rest_len[i] = (p1 - p2).norm()

    @staticmethod
    @ti.kernel
    def init_pos(
        pos:ti.template(),
        N:ti.i32,
    ):
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            # pos[idx] = ti.Vector([i / N,  j / N, 0.5])  # vertical hang
            pos[idx] = ti.Vector([i / N, 0.5, j / N]) # horizontal hang


    @staticmethod
    @ti.kernel
    def init_mass(
        inv_mass:ti.template(),
        N:ti.i32,
        NV:ti.i32,
        setup_num:ti.i32,
    ):
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            inv_mass[idx] = 1.0
        if setup_num == 0: # fix point
            inv_mass[N] = 0.0
            inv_mass[NV-1] = 0.0

    @staticmethod
    @ti.kernel
    def init_tri(tri:ti.template(), N:ti.i32):
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @staticmethod
    @ti.kernel
    def init_edge(
        edge:ti.template(),
        N:ti.i32,
    ):
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])





def write_and_rebuild_topology(edge:np.ndarray, tri:np.ndarray, out_dir:str):
    """
    write topology to file and rebuild the topology mapping 

    vertex to edge, vertex to tri, edge to tri
    """
    from engine.mesh_io import write_edge, write_tri
    from engine.mesh_io import build_vertex2edge, build_vertex2tri, build_edge2tri

    assert tri.shape[1] == 3
    assert edge.shape[1] == 2

    write_edge(out_dir + f"/mesh/edge", edge)
    write_tri(out_dir + f"/mesh/tri", tri)

    # rebuild mapping
    v2e = build_vertex2edge(edge) #dict vertex to edge
    v2t = build_vertex2tri(tri)   #dict vertex to tri
    e2t = build_edge2tri(edge,v2t,tri) #dict edge to tri

    # check topology
    e = np.random.randint(edge.shape[0])
    v0,v1 = edge[e]
    assert e in v2e[v0] #check v2e
    for t in e2t[e]:    #check e2t
        assert v0 in tri[t] and v1 in tri[t]
    for t in v2t[v0]:   #check v2t
        assert v0 in tri[t]

    # write to json
    import json

    def convert_keys_to_str(d):
        """递归地将字典的键转换为字符串"""
        if isinstance(d, dict):
            return {str(k): convert_keys_to_str(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_keys_to_str(i) for i in d]
        else:
            return d

    v2e_s = convert_keys_to_str(v2e)
    v2t_s = convert_keys_to_str(v2t)
    e2t_s = convert_keys_to_str(e2t)
    with open(out_dir + f"/mesh/v2e.json", "w") as f:
        s = json.dumps(v2e_s, indent=4)
        f.write(s)
    with open(out_dir + f"/mesh/v2t.json", "w") as f:
        s = json.dumps(v2t_s, indent=4)
        f.write(s)
    with open(out_dir + f"/mesh/e2t.json", "w") as f:
        s = json.dumps(e2t_s, indent=4)
        f.write(s)
    return v2e, v2t, e2t



