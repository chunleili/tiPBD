from visualize import visualize
import taichi as ti


def read_particles(mesh_path="data/model/bunny.obj"):
    import trimesh
    print("Read ", mesh_path)
    mesh = trimesh.load(mesh_path)
    return mesh.vertices

# def read_particles_meshio(mesh_path="data/model/bunny.obj"):
#     import meshio
#     print("Read ", mesh_path)
#     mesh = meshio.read(mesh_path)
#     return mesh.points

def read_tetgen(filename):
    '''
    读取tetgen生成的网格文件，返回顶点坐标、单元索引、面索引

    Args:
        filename: 网格文件名，不包含后缀名
    
    Returns:
        pos: 顶点坐标，shape=(NV, 3)
        tet_indices: 单元索引，shape=(NT, 4)
        face_indices: 面索引，shape=(NF, 3)
    '''
    import numpy as np
    ele_file_name = filename + '.ele'
    node_file_name = filename + '.node'
    face_file_name = filename + '.face'

    with open(node_file_name, 'r') as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(ele_file_name, 'r') as f:
        lines = f.readlines()
        NT = int(lines[0].split()[0])
        tet_indices = np.zeros((NT, 4), dtype=np.int32)
        for i in range(NT):
            tet_indices[i] = np.array(lines[i + 1].split()[1:], dtype=np.int32)

    with open(face_file_name, 'r') as f:
        lines = f.readlines()
        NF = int(lines[0].split()[0])
        face_indices = np.zeros((NF, 3), dtype=np.int32)
        for i in range(NF):
            face_indices[i] = np.array(lines[i + 1].split()[1:-1],
                                       dtype=np.int32)

    return pos, tet_indices, face_indices


def points_from_volume(mesh_path="data/model/box.obj", particle_seperation=0.02):
    '''
    将surface mesh转换为体素化后的点云（粒子化）。点云的采样密度由particle_seperation决定。这与houdini中的
    points_from_volume节点一致。

    Args:
        mesh_path: mesh文件路径
        particle_seperation: 粒子间距
    
    Returns:
        mesh_vox_pts: 均匀采样的点云
    '''
    import trimesh

    mesh = trimesh.load(mesh_path)

    mesh_vox = mesh.voxelized(pitch=particle_seperation).fill()
    point_cloud = mesh_vox.points
    return point_cloud


def scale_to_unit_sphere(mesh):
    '''
    将mesh缩放到单位球，并且将mesh的中心点移动到原点。

    Args:
        mesh: 原始Trimesh对象

    Returns:
        缩放后的Trimesh对象
    '''
    import trimesh
    import numpy as np
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    '''
    将mesh缩放到单位立方体。并且将mesh的中心点移动到原点。

    Args:
        mesh: 原始Trimesh对象

    Returns:
        缩放后的Trimesh对象
    '''
    import trimesh
    import numpy as np

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #


def test_points_from_volume():
    pts_np = points_from_volume()
    import taichi as ti
    ti.init()
    pts = ti.Vector.field(3, dtype=ti.f32, shape=pts_np.shape[0])
    pts.from_numpy(pts_np)
    from visualize import visualize
    visualize(pts)


def test_scale_to_unit_sphere():
    import trimesh
    mesh = trimesh.load("data/model/bunny.obj")
    print("before scale")
    mesh = scale_to_unit_sphere(mesh)
    print("after scale")
    visualize(mesh.vertices, ti_init=True)

def test_scale_to_unit_cube():
    import trimesh
    mesh = trimesh.load("data/model/bunny.obj")
    print("before scale")
    print(mesh.vertices.max(), mesh.vertices.min())
    mesh = scale_to_unit_cube(mesh)
    print("after scale")
    print(mesh.vertices.max(), mesh.vertices.min())
    visualize(mesh.vertices, ti_init=True)


def test_read_particles():
    pts = read_particles()
    visualize(pts, ti_init=True, show_widget=True)


if __name__=="__main__":
    # test_scale_to_unit_cube()
    # test_points_from_volume()
    # test_read_particles()
    test_read_particles()