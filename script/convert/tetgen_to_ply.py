
import sys,os
import meshio
from pathlib import Path
import numpy as np

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

def read_tet(filename, build_face_flag=False):
    mesh = meshio.read(filename)
    pos = mesh.points
    tet_indices = mesh.cells_dict["tetra"]
    if build_face_flag:
        face_indices = build_face_indices(tet_indices)
        return pos, tet_indices, face_indices
    else:
        return pos, tet_indices

def tetgen_to_ply(mesh_path):
    pos, tet_indices, face_indices = read_tet(mesh_path+".node", build_face_flag=True)
    mesh = meshio.Mesh(pos, {"tetra": tet_indices, "triangle": face_indices})
    mesh.write(mesh_path+".ply")
    print("Write to: ", Path(mesh_path+".ply"))

if __name__ == "__main__":
    mesh_path = "D:/dev/PainlessMG/CUDA_Projective_Armadillo/meshes/squirrel_modified_big"
    tetgen_to_ply(mesh_path)
    print("Done.")