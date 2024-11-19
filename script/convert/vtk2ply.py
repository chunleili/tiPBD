"""
vtk to ply with user properties
"""

import meshio
import plyfile
from plyfile import PlyData, PlyElement
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Convert VTK to PLY')
parser.add_argument('-input', type=str, help='Input VTK file')
args = parser.parse_args()

    
def vtk2ply(input,output=None):
    # Read the VTK file
    mesh = meshio.read(input)

    # Extract the vertices and faces
    vertices = mesh.points
    strain = mesh.cell_data['strain'][0]
    tri = mesh.cells_dict['triangle']

    # Create a structured array for faces
    face_dtype = [('vertex_indices', 'int32', (3,)), ('strain', strain.dtype, (3,))]
    faces = np.empty(len(tri), dtype=face_dtype)
    faces['vertex_indices'] = tri
    faces['strain'] = strain

    # Convert vertices to a structured array
    vertex_dtype = [('x', vertices.dtype), ('y', vertices.dtype), ('z', vertices.dtype)]
    vertices_structured = np.array([tuple(v) for v in vertices], dtype=vertex_dtype)

    # Create a PLY file
    ply = plyfile.PlyData([
        plyfile.PlyElement.describe(vertices_structured, 'vertex'),
        plyfile.PlyElement.describe(faces, 'face'),
    ])

    if output is None:
        write_path = input.replace('.vtk', '.ply')
    else:
        write_path = output

    # with open(write_path, mode='w') as f:
    #     ply.write(f)
    ply.text = True
    ply.write(write_path)
    # ply.write(write_path, txt=True)
    print(f'PLY file saved to {write_path}')
    return ply


if __name__ == '__main__':
    vtk2ply("E:/Dev/tiPBD/result/latest1/mesh/0001.vtk","test.ply")