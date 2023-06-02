import taichi as ti
import meshio


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
    import numpy as np

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    mesh.vertices = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3].dot(mesh.vertices.T).T


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
