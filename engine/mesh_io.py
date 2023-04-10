def read_particles(filepath):
    import meshio
    print("read ", filepath)
    mesh = meshio.read(filepath)
    return mesh.points


def read_tetgen(filename):
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


def point_cloud_from_mesh(mesh_path="data/model/box.obj", particle_seperation=0.02):
    '''
    将surface mesh转换为点云
    
    Args:
        mesh_path: mesh文件路径
        particle_seperation: 粒子间距
    
    Returns:
        mesh_vox_pts: 粒子化点云
    '''
    import trimesh

    mesh = trimesh.load(mesh_path)

    mesh_vox = mesh.voxelized(pitch=particle_seperation).fill()
    mesh_vox_pts = mesh_vox.points

    return mesh_vox_pts


if __name__=="__main__":
    pts_np = point_cloud_from_mesh()
    import taichi as ti
    ti.init()
    pts = ti.Vector.field(3, dtype=ti.f32, shape=pts_np.shape[0])
    pts.from_numpy(pts_np)
    from solver_main import visualize
    visualize(pts)