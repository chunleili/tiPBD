
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