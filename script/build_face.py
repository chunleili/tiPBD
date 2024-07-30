import meshio
import numpy as np

def read_tet(filename, build_face_flag=False):
    mesh = meshio.read(filename)
    pos = mesh.points
    tet_indices = mesh.cells_dict["tetra"]
    if build_face_flag:
        face_indices = build_face_indices(tet_indices)
        return pos, tet_indices, face_indices
    else:
        return pos, tet_indices

def build_face_indices(tet_indices):
    face_indices = np.empty((tet_indices.shape[0] * 4, 3), dtype=np.int32)
    for t in range(tet_indices.shape[0]):
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in range(4):  # 4 faces
            for j in range(3):  # 3 vertices
                face_indices[t * 4 + i][j] = tet_indices[t][ind[i][j]]
    return face_indices


# write .node file
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


p,t,f = read_tet("data/model/bunnyBig/bunnyBig.node", build_face_flag=True)
write_tetgen("data/model/bunnyBig/bunnyBig1", p, t, f)