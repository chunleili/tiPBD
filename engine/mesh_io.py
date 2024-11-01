import taichi as ti
import meshio
import numpy as np





def read_particles(mesh_path="data/model/bunny.obj"):
    import trimesh

    print("Read ", mesh_path)
    mesh = trimesh.load(mesh_path)
    return mesh.vertices


def read_meshio_instance(mesh_path="data/model/bunny.obj"):
    import meshio

    print("Using meshio read: ", mesh_path)
    mesh = meshio.read(mesh_path)
    return mesh


def read_trimesh_instance(mesh_path="data/model/bunny.obj"):
    import trimesh

    print("Using trimesh read: ", mesh_path)
    mesh = mesh = trimesh.load(mesh_path)
    return mesh


def read_mesh(mesh_path="data/model/bunny.obj", scale=[1.0, 1.0, 1.0], shift=[0, 0, 0]):
    import trimesh

    print("Using trimesh read ", mesh_path)
    mesh = trimesh.load(mesh_path)
    mesh.vertices *= scale
    mesh.vertices += shift
    return mesh.vertices, mesh.faces


def write_tet(filename, points, tet_indices):
    cells = [
        ("tetra", tet_indices),
    ]
    mesh = meshio.Mesh(
        points,
        cells,
    )
    mesh.write(filename)
    return mesh


def write_obj(filename, pos, tri):
    """
    example: write_obj(out_dir + "cloth.obj", pos.to_numpy(), tri.to_numpy())
    """
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )
    mesh.write(filename)
    return mesh

def tetgen_to_ply(mesh_path):
    print("Using meshio to read: ", mesh_path + ".node")
    mesh = meshio.read(mesh_path + ".node", file_format="tetgen")
    mesh.write(mesh_path + ".ply", binary=False)
    print("Write to: ", mesh_path + ".ply")


def tetgen_to_vtk(mesh_path):
    print("Using meshio to read: ", mesh_path + ".node")
    mesh = meshio.read(mesh_path + ".node", file_format="tetgen")
    mesh.write(mesh_path + ".vtk")
    print("Write to: ", mesh_path + ".vtk")


def read_tetgen(filename):
    """
    读取tetgen生成的网格文件，返回顶点坐标、单元索引、面索引

    Args:
        filename: 网格文件名，不包含后缀名

    Returns:
        pos: 顶点坐标，shape=(NV, 3)
        tet_indices: 单元索引，shape=(NT, 4)
        face_indices: 面索引，shape=(NF, 3)
    """
    import numpy as np

    ele_file_name = filename + ".ele"
    node_file_name = filename + ".node"
    face_file_name = filename + ".face"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(ele_file_name, "r") as f:
        lines = f.readlines()
        NT = int(lines[0].split()[0])
        tet_indices = np.zeros((NT, 4), dtype=np.int32)
        for i in range(NT):
            tet_indices[i] = np.array(lines[i + 1].split()[1:], dtype=np.int32)

    with open(face_file_name, "r") as f:
        lines = f.readlines()
        NF = int(lines[0].split()[0])
        face_indices = np.zeros((NF, 3), dtype=np.int32)
        for i in range(NF):
            face_indices[i] = np.array(lines[i + 1].split()[1:-1], dtype=np.int32)

    return pos, tet_indices, face_indices


def write_tetgen(filename, points, tet_indices, tri_indices=None):
    node_mesh = filename + ".node"
    ele_mesh = filename + ".ele"
    face_mesh = filename + ".face"
    with open(node_mesh, "w") as f:
        f.write(f"{points.shape[0]} 3 0 0\n")
        for i in range(points.shape[0]):
            f.write(f"{i} {points[i, 0]} {points[i, 1]} {points[i, 2]}\n")
    with open(ele_mesh, "w") as f:
        f.write(f"{tet_indices.shape[0]} 4 0\n")
        for i in range(tet_indices.shape[0]):
            f.write(f"{i} {tet_indices[i, 0]} {tet_indices[i, 1]} {tet_indices[i, 2]} {tet_indices[i, 3]}\n")
    if tri_indices is not None:
        with open(face_mesh, "w") as f:
            f.write(f"{tri_indices.shape[0]} 3 0\n")
            for i in range(tri_indices.shape[0]):
                f.write(f"{i} {tri_indices[i, 0]} {tri_indices[i, 1]} {tri_indices[i, 2]} -1\n")


def points_from_volume(mesh_path="data/model/box.obj", particle_seperation=0.02):
    """
    将surface mesh转换为体素化后的点云（粒子化）。点云的采样密度由particle_seperation决定。这与houdini中的
    points_from_volume节点一致。

    Args:
        mesh_path: mesh文件路径
        particle_seperation: 粒子间距

    Returns:
        mesh_vox_pts: 均匀采样的点云
    """
    import trimesh

    mesh = trimesh.load(mesh_path)

    mesh_vox = mesh.voxelized(pitch=particle_seperation).fill()
    point_cloud = mesh_vox.points
    return point_cloud


def scale_to_unit_sphere(mesh):
    """
    将mesh缩放到单位球，并且将mesh的中心点移动到原点。

    Args:
        mesh: 原始Trimesh对象

    Returns:
        缩放后的Trimesh对象
    """
    import trimesh
    import numpy as np

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def scale_to_unit_cube(mesh):
    """
    将mesh缩放到[-1,1]的立方体，并且将mesh的中心点移动到原点。

    Args:
        mesh: 原始Trimesh对象

    Returns:
        缩放后的Trimesh对象
    """
    import trimesh
    import numpy as np

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def shift(mesh, x):
    mesh.vertices += x


def scale(mesh, s):
    mesh.vertices *= s


def rotate(mesh, axis, angle):
    import trimesh

    # if isinstance(mesh, trimesh.Scene):
    #     mesh = mesh.dump().sum()
    # mesh.vertices = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3].dot(mesh.vertices.T).T

    t = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(t)



def match_size(mesh, bbox):
    """
    将mesh缩放到bbox的大小，并且将mesh的中心点移动到bbox的中心点。这与Houdini中的match_size节点一致。

    Args:
        mesh: Trimesh对象
        bbox: 目标bbox，格式为[[xmin, ymin, zmin], [xmax, ymax, zmax]]
    """
    bbox_extents = bbox[1][:] - bbox[0][:]
    bbox_centroid = (bbox[1][:] + bbox[0][:]) * 0.5
    mesh.vertices *= bbox_extents / mesh.bounding_box.extents
    mesh.vertices += bbox_centroid - mesh.bounding_box.centroid


@ti.kernel
def translation_ti(pos: ti.template(), tx: ti.f32, ty: ti.f32, tz: ti.f32):
    for i in pos:
        pos[i] = pos[i] + ti.Vector([tx, ty, tz])


@ti.kernel
def scale_ti(pos: ti.template(), sx: ti.f32, sy: ti.f32, sz: ti.f32):
    for i in pos:
        pos[i] = pos[i] * ti.Vector([sx, sy, sz])


def build_face_indices(tet_indices):
    """
    从四面体索引构建面索引
    """
    face_indices = np.empty((tet_indices.shape[0] * 4, 3), dtype=np.int32)
    for t in range(tet_indices.shape[0]):
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in range(4):  # 4 faces
            for j in range(3):  # 3 vertices
                face_indices[t * 4 + i][j] = tet_indices[t][ind[i][j]]
    return face_indices


@ti.kernel
def tet_indices_to_face_indices(tets: ti.template(), faces: ti.template()):
    for tid in tets:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        faces[tid * 4 + 0][0] = tets[tid][ind[0][0]]
        faces[tid * 4 + 0][1] = tets[tid][ind[0][1]]
        faces[tid * 4 + 0][2] = tets[tid][ind[0][2]]
        faces[tid * 4 + 1][0] = tets[tid][ind[1][0]]
        faces[tid * 4 + 1][1] = tets[tid][ind[1][1]]
        faces[tid * 4 + 1][2] = tets[tid][ind[1][2]]
        faces[tid * 4 + 2][0] = tets[tid][ind[2][0]]
        faces[tid * 4 + 2][1] = tets[tid][ind[2][1]]
        faces[tid * 4 + 2][2] = tets[tid][ind[2][2]]
        faces[tid * 4 + 3][0] = tets[tid][ind[3][0]]
        faces[tid * 4 + 3][1] = tets[tid][ind[3][1]]
        faces[tid * 4 + 3][2] = tets[tid][ind[3][2]]


@ti.kernel
def tet_indices_to_edge_indices(tets: ti.template(), edges: ti.template()):
    ind = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for tid in tets:
        edges[tid * 6 + 0][0] = tets[tid][ind[0][0]]
        edges[tid * 6 + 0][1] = tets[tid][ind[0][1]]
        edges[tid * 6 + 1][0] = tets[tid][ind[1][0]]
        edges[tid * 6 + 1][1] = tets[tid][ind[1][1]]
        edges[tid * 6 + 2][0] = tets[tid][ind[2][0]]
        edges[tid * 6 + 2][1] = tets[tid][ind[2][1]]
        edges[tid * 6 + 3][0] = tets[tid][ind[3][0]]
        edges[tid * 6 + 3][1] = tets[tid][ind[3][1]]
        edges[tid * 6 + 4][0] = tets[tid][ind[4][0]]
        edges[tid * 6 + 4][1] = tets[tid][ind[4][1]]
        edges[tid * 6 + 5][0] = tets[tid][ind[5][0]]
        edges[tid * 6 + 5][1] = tets[tid][ind[5][1]]


def make_no_ext(filename):
    return filename.split(".")[0]




# ---------------------------------------------------------------------------- #
#                                      new                                     #
# ---------------------------------------------------------------------------- #

def write_mesh(filename, pos, tri, format="ply"):
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )

    if format == "ply":
        mesh.write(filename + ".ply", binary=True)
    elif format == "obj":
        mesh.write(filename + ".obj")
    else:
        raise ValueError("Unknown format")
    return mesh


def write_edge(filename, data):
    np.savetxt(filename + ".txt", data, fmt="%d")

def write_tri(filename, data):
    np.savetxt(filename + ".txt", data, fmt="%d")


# TODO: only vtk support for now
def write_vtk_with_strain(filename, pos, tri, **kwargs):
    binary = kwargs.get("binary", True)
    strain = kwargs.get("strain", None)
    if strain is None:
        raise ValueError("strain data is required")
    cells = [("triangle", tri.reshape(-1, 3)),]
    cell_data = {"strain": [strain]}
    mesh = meshio.Mesh(pos, cells, cell_data=cell_data)
    mesh.write(filename + ".vtk", binary=binary)
    return mesh


def write_ply_with_strain(filename, pos, tri, strain, binary=False):
    import plyfile
    # meshio do not support writing user properties to ply, so we use plyfile
    
    # Create a structured array for faces
    face_dtype = [('vertex_indices', 'int32', (3,)), ('strain', strain.dtype)]
    faces = np.empty(len(tri), dtype=face_dtype)
    faces['vertex_indices'] = tri
    faces['strain'] = strain

    # Convert pos to a structured array
    vertex_dtype = [('x', pos.dtype), ('y', pos.dtype), ('z', pos.dtype)]
    vertices_structured = np.array([tuple(v) for v in pos], dtype=vertex_dtype)

    # Create a PLY file
    ply = plyfile.PlyData([
        plyfile.PlyElement.describe(vertices_structured, 'vertex'),
        plyfile.PlyElement.describe(faces, 'face'),
    ])

    ply.text = not binary
    filename = filename + ".ply"
    ply.write(filename)
    print(f'PLY file saved to {filename}')



def write_edge_data(filename, data):
    """ 
    Write data stored in edges to file
    e.g. write the strain of cloth
    """
    with open(filename+".txt", "w") as f:
        f.write(f"edge shape={data.shape} dtype={data.dtype}\n")
        np.savetxt(f, data)


def read_tet(filename, build_face_flag=False):
    mesh = meshio.read(filename)
    pos = mesh.points
    tet_indices = mesh.cells_dict["tetra"]
    if build_face_flag:
        face_indices = build_face_indices(tet_indices)
        return pos, tet_indices, face_indices
    else:
        return pos, tet_indices



# Usage:
# # original size: 0.1 and 0.5
# coarse_points, coarse_tet_indices, coarse_tri_indices = generate_cube_mesh(1.0, 1.0)
# write_tet("data/model/cube/minicube.node", coarse_points, coarse_tet_indices)
def generate_cube_mesh(len, grid_dx=0.1):
    num_grid = int(len // grid_dx)
    points = np.zeros(((num_grid + 1) ** 3, 3), dtype=float)
    for i in range(num_grid + 1):
        for j in range(num_grid + 1):
            for k in range(num_grid + 1):
                points[i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k] = [i * grid_dx, j * grid_dx, k * grid_dx]
    tet_indices = np.zeros(((num_grid) ** 3 * 5, 4), dtype=int)
    tri_indices = np.zeros(((num_grid) ** 3 * 12, 3), dtype=int)
    for i in range(num_grid):
        for j in range(num_grid):
            for k in range(num_grid):
                id0 = i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k
                id1 = i * (num_grid + 1) ** 2 + j * (num_grid + 1) + k + 1
                id2 = i * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k
                id3 = i * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k + 1
                id4 = (i + 1) * (num_grid + 1) ** 2 + j * (num_grid + 1) + k
                id5 = (i + 1) * (num_grid + 1) ** 2 + j * (num_grid + 1) + k + 1
                id6 = (i + 1) * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k
                id7 = (i + 1) * (num_grid + 1) ** 2 + (j + 1) * (num_grid + 1) + k + 1
                tet_start = (i * num_grid**2 + j * num_grid + k) * 5
                tet_indices[tet_start] = [id0, id1, id2, id4]
                tet_indices[tet_start + 1] = [id1, id4, id5, id7]
                tet_indices[tet_start + 2] = [id2, id4, id6, id7]
                tet_indices[tet_start + 3] = [id1, id2, id3, id7]
                tet_indices[tet_start + 4] = [id1, id2, id4, id7]
                tri_start = (i * num_grid**2 + j * num_grid + k) * 12
                tri_indices[tri_start] = [id0, id2, id4]
                tri_indices[tri_start + 1] = [id2, id4, id6]
                tri_indices[tri_start + 2] = [id0, id1, id2]
                tri_indices[tri_start + 3] = [id1, id2, id3]
                tri_indices[tri_start + 4] = [id0, id1, id4]
                tri_indices[tri_start + 5] = [id1, id4, id5]
                tri_indices[tri_start + 6] = [id4, id5, id6]
                tri_indices[tri_start + 7] = [id3, id6, id7]
                tri_indices[tri_start + 8] = [id4, id5, id6]
                tri_indices[tri_start + 9] = [id5, id6, id7]
                tri_indices[tri_start + 10] = [id1, id3, id7]
                tri_indices[tri_start + 11] = [id1, id5, id7]
    write_tet("data/model/cube/coarse_new.node", points, tet_indices)
    return points, tet_indices, tri_indices







def read_tri_cloth(filename):
    edge_file_name = filename + ".edge"
    node_file_name = filename + ".node"
    face_file_name = filename + ".face"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(edge_file_name, "r") as f:
        lines = f.readlines()
        NE = int(lines[0].split()[0])
        edge_indices = np.zeros((NE, 2), dtype=np.int32)
        for i in range(NE):
            edge_indices[i] = np.array(lines[i + 1].split()[1:], dtype=np.int32)

    with open(face_file_name, "r") as f:
        lines = f.readlines()
        NF = int(lines[0].split()[0])
        face_indices = np.zeros((NF, 3), dtype=np.int32)
        for i in range(NF):
            face_indices[i] = np.array(lines[i + 1].split()[1:-1], dtype=np.int32)

    return pos, edge_indices, face_indices.flatten(), NE, NV


def read_tri_cloth_obj(path):
    print(f"path is {path}")
    mesh = meshio.read(path)
    tri = mesh.cells_dict["triangle"]
    pos = mesh.points

    num_tri = len(tri)
    edges=[]
    for i in range(num_tri):
        ele = tri[i]
        edges.append([min((ele[0]), (ele[1])), max((ele[0]),(ele[1]))])
        edges.append([min((ele[1]), (ele[2])), max((ele[1]),(ele[2]))])
        edges.append([min((ele[0]), (ele[2])), max((ele[0]),(ele[2]))])
    #remove the duplicate edges
    # https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
    import itertools
    edges.sort()
    edges = list(edges for edges,_ in itertools.groupby(edges))

    return pos, np.array(edges), tri.flatten()


def set_to_list(s):
    for k, v in s.items():
        s[k] = list(v)
    return s


def build_vertex2edge(edges: np.ndarray)->dict:
    v2e = {} #vertex to edge
    for edge_index, (v1, v2) in enumerate(edges):
        if v1 not in v2e:
            v2e[v1] = set()
        if v2 not in v2e:
            v2e[v2] = set()
        v2e[v1].add(edge_index)
        v2e[v2].add(edge_index)
    
    for k, v in v2e.items():
        v2e[k] = list(v)
    return v2e


def build_vertex2tri(tri: np.ndarray)->dict:
    assert tri.shape[1] == 3
    v2t = {} #vertex to triangle
    for tri_index, (v0, v1, v2) in enumerate(tri):
        if v0 not in v2t:
            v2t[v0] = set()
        if v1 not in v2t:
            v2t[v1] = set()
        if v2 not in v2t:
            v2t[v2] = set()
        v2t[v0].add(tri_index)
        v2t[v1].add(tri_index)
        v2t[v2].add(tri_index)
    
    for k, v in v2t.items():
        v2t[k] = list(v)
    return v2t


def build_edge2tri(edge: np.ndarray, v2t: dict, tri:np.ndarray)->dict:
    """
        Args:
            edge: shape=(NE, 2)
            v2t: vertex to triangle mapping (dict of list)
        Returns:
            e2t: edge to triangle mapping
        Note: First call build_vertex2tri to get v2t
    """    
    assert edge.shape[1] == 2
    assert tri.shape[1] == 3
    e2t = {} #edge to triangle
    for e in range(edge.shape[0]):
        v0, v1 = edge[e]
        tris0 = v2t[v0]
        tris1 = v2t[v1]
        # If a triangle has both v0 and v1, then it is an edge
        tris = tris0 + tris1
        for t in tris:
            if v0 in tri[t] and v1 in tri[t]:
                if e not in e2t:
                    e2t[e] = set()
                e2t[e].add(t)
        
    for k, v in e2t.items():
        e2t[k] = list(v)
    return e2t



def edge_data_to_tri_data(e2t, edge_data, tri):
    tri_data = np.zeros((tri.shape[0]))
    NE = edge_data.shape[0]
    for e in range(NE):
        tris = e2t[e]
        for t in tris: # triangles that has edge e
            # TODO: now we use sum square of edge data into one triangle data to get a scalar value, maybe we can use vec3
            tri_data[t] += edge_data[e]**2
    return tri_data