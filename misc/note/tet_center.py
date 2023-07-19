import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init()


@ti.func
def tet_centroid_func(p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0])
    pc_p0 = A @ ti.math.vec3([0.5, 0.5, 0.5])
    pc = pc_p0 + p0
    return pc


vec3 = ti.math.vec3
ivec4 = ti.math.ivec4


@ti.kernel
def compute_tet_centroid(
    pos: ti.types.ndarray(dtype=vec3),
    tet_indices: ti.types.ndarray(dtype=ivec4),
    tet_centroid: ti.types.ndarray(dtype=vec3),
):
    for t in range(tet_indices.shape[0]):
        a, b, c, d = tet_indices[t]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        tet_centroid[t] = tet_centroid_func(p0, p1, p2, p3)


# pos = np.array([[0,0,0],
#                 [1,0,0],
#                 [0.5,0.86,0],
#                 [0.5,0.43,0.86]])
pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]])

tet_indices = np.array([[0, 1, 2, 3]])
tet_centroid = np.zeros((1, 3))
compute_tet_centroid(pos, tet_indices, tet_centroid)

c = np.mean(pos, axis=0)
print("c", c)
print("tet_centroid", tet_centroid)


ax = plt.axes(projection="3d")
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="b")
ax.scatter(c[0], c[1], c[2], c="r")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
