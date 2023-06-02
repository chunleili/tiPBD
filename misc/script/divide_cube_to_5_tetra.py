import numpy as np
import meshio


def divide_cube_to_5_tetra(filename):
    points = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )

    tet_indices = np.array([[0, 3, 5, 6], [0, 1, 3, 5], [0, 2, 3, 6], [0, 4, 5, 6], [3, 5, 6, 7]])

    tri_indices = np.array(
        [
            [0, 5, 6],
            [0, 3, 5],
            [0, 6, 3],
            [3, 6, 5],
            [0, 3, 1],
            [0, 5, 3],
            [0, 1, 5],
            [1, 3, 5],
            [0, 2, 3],
            [0, 3, 6],
            [0, 6, 2],
            [2, 6, 3],
            [0, 5, 4],
            [0, 4, 6],
            [0, 6, 5],
            [4, 5, 6],
            [3, 6, 5],
            [3, 7, 5],
            [3, 6, 7],
            [5, 7, 6],
        ]
    )

    cells = [
        ("tetra", tet_indices),
        ("triangle", tri_indices),
    ]

    mesh = meshio.Mesh(
        points,
        cells,
    )
    mesh.write(filename)
    return mesh


if __name__ == "__main__":
    divide_cube_to_5_tetra("cube_5_tetra.ply")
    divide_cube_to_5_tetra("cube_5_tetra.vtk")
    divide_cube_to_5_tetra("cube_5_tetra.node")
