"""
use case:

if args.use_bending:
    self.tri_pairs, self.bending_length = init_bending(self.tri, self.pos)
    self.vert, self.NCONS = add_distance_constraints_from_tri_pairs(self.vert, self.tri_pairs)

"""


import numpy as np
from time import perf_counter
import taichi as ti


# tri:  (num_tri, 3)
# https://matthias-research.github.io/pages/tenMinutePhysics/14-cloth.pdf last page
# https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/14-cloth.html
def find_tri_neighbors(tri):
    # 1. Build the edge list
    num_tri = tri.shape[0]
    # This is different with existing edge. 
    # This new edge list may have duplicates because in different triangles.
    edge_list = [] # [v0, v1, gid]
    # gid = 3 * tid + 0/1/2(local id)
    for t in range(num_tri):
        v0 = tri[t, 0]
        v1 = tri[t, 1]
        v2 = tri[t, 2]
        # [0,1][1,2][2,0]: the counter-clockwise order, we use this
        # [0,2][2,1][1,0]: the clockwise order, also reasonable
        edge_list.append([min(v0, v1), max(v0, v1), 3 * t + 0])
        edge_list.append([min(v1, v2), max(v1, v2), 3 * t + 1])
        edge_list.append([min(v2, v0), max(v2, v0), 3 * t + 2])

    # 2. Sort the edge list by the two vertices
    edge_list = np.array(edge_list)
    sorted_indices = np.lexsort((edge_list[:, 1], edge_list[:, 0]))
    sorted = edge_list[sorted_indices]


    # 3. Find the tri neighbors by duplicated edges
    # tri_neighbor: (num_tri, 3), each row is the gid of the neighbor triangle. We could also use tid instead of gid, but gid gives more information. gid//3 is the tid(neigbour triangle), gid%3 is the local edge id(which edge of the  neighbor triangle)
    tri_neighbor = np.ones((num_tri, 3), dtype=np.int32) * (-1)
    for i in range(0, len(sorted), 2):
        # If the first 2 values of sorted edge list is the same, then  they are the same edge with different triangles. Then these two triangles are neighbors.
        if i + 1 < len(sorted) and \
        (sorted[i][0] == sorted[i + 1][0]) and \
        (sorted[i][1] == sorted[i + 1][1]):
            gid0 = sorted[i, 2]
            gid1 = sorted[i + 1, 2]
            tid0 = gid0 // 3     # triangle id
            tid1 = gid1 // 3 
            localid0 = gid0 % 3  # which edge(0/1/2)
            localid1 = gid1 % 3
            # CAUTION: We store gid instead of tid
            tri_neighbor[tid0, localid0] = gid1  
            tri_neighbor[tid1, localid1] = gid0
    return tri_neighbor


def build_tri_pairs(tri, tri_neighbor):
    num_tri = tri.shape[0]
    tri_pairs = []
    for i in range(num_tri): # 遍历三角形
        for j in range(3):   # 遍历三角形的三个边
            gid = tri_neighbor[i, j] #三角形i的第j条边的邻居global edge id，如果没有邻居则为-1。 
            # gid = 3 * tri_id + 0/1/2 
            # 因此gid % 3 得到local edge id, 即共享边是邻居三角形的第几个边
            # gid // 3 得到tri_id，即邻居三角形的id
            if gid >= 0: # 有邻居, 即不是-1
                tid = gid // 3      # 邻居三角形的id
                localid = gid % 3   # 邻居三角形的第几个边
                id0 = tri[i, j]             # 三角形i的第0个点
                id1 = tri[i, (j + 1) % 3]   # 三角形i的第1个点
                id2 = tri[i, (j + 2) % 3]   # 三角形i的第2个点
                id3 = tri[tid, (localid + 2) % 3] # 邻居三角形的非共享点, 即它的第三个点
                tri_pairs.append([id0, id1, id2, id3])
    return tri_pairs


def init_bending_length(tri_pairs, pos):
    bending_id = tri_pairs.copy()
    bending_length = np.zeros(len(bending_id), dtype=np.float32)
    for i in range(bending_length.shape[0]):
        v2 = bending_id[i, 2]
        v3 = bending_id[i, 3]
        bending_length[i] = np.linalg.norm(pos[v2] - pos[v3])
    return bending_length


@ti.kernel
def init_bending_length_kernel(tri_pairs:ti.types.ndarray(), pos:ti.template(), bending_length:ti.types.ndarray()):
    for i in range((tri_pairs).shape[0]):
        v2 = tri_pairs[i, 2]
        v3 = tri_pairs[i, 3]
        bending_length[i] = (pos[v2] - pos[v3]).norm()


def init_bending(tri, pos):
    print("init_bending...")
    tic = perf_counter()
    tic1 = perf_counter()
    tri_neighbor = find_tri_neighbors(tri)
    # print("邻居边编号列表:", tri_neighbor)
    print(f"find_tri_neighbors time: {perf_counter() - tic1}")

    # tri_pairs有四个点，第四个点是另一个三角形的点
    tic2 = perf_counter()
    tri_pairs = build_tri_pairs(tri, tri_neighbor)
    tri_pairs = np.array(tri_pairs, dtype=np.int32)
    # print("三角形对列表:", tri_pairs)
    print(f"build_tri_pairs time: {perf_counter() - tic2}")

    tic3 = perf_counter()
    # bending_length = init_bending_length(tri_pairs, pos.to_numpy())
    bending_length = np.zeros(len(tri_pairs), dtype=np.float32)
    init_bending_length_kernel(tri_pairs, pos, bending_length)
    # print("弯曲长度列表:", bending_length)
    print(f"init_bending_length time: {perf_counter() - tic3}")
    print(f"init_bending time: {perf_counter() - tic}")
    return tri_pairs, bending_length




@ti.kernel
def solve_bending_constraints_xpbd(
    dual_residual_bending: ti.template(),
    inv_mass:ti.template(),
    lagrangian_bending:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    bending_length:ti.types.ndarray(),
    tri_pairs:ti.types.ndarray(),
    alpha_bending:ti.f32
):
    for i in range(bending_length.shape[0]):
        idx0, idx1 = tri_pairs[i, 2], tri_pairs[i, 3]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        if invM0 == 0.0 and invM1 == 0.0:
            continue
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - bending_length[i]
        gradient = dis.normalized()
        if gradient.norm() == 0.0:
            continue
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian_bending[i] * alpha_bending) / (invM0 + invM1 + alpha_bending)
        lagrangian_bending[i] += delta_lagrangian

        # residual
        dual_residual_bending[i] = (constraint + alpha_bending * lagrangian_bending[i])
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient




def add_distance_constraints_from_tri_pairs(old_vert, tri_pairs):
    old_NCONS = old_vert.shape[0]
    added_NCONS = tri_pairs.shape[0]
    new_NCONS = added_NCONS + old_NCONS
    vert = ti.Vector.field(2, dtype=ti.i32, shape= new_NCONS)
    @ti.kernel
    def kernel(vert:ti.template(), 
                old_vert:ti.template(), 
                tri_pairs:ti.types.ndarray()):
            for i in range(old_vert.shape[0]):
                vert[i] = old_vert[i]

            for i in range(old_vert.shape[0], tri_pairs.shape[0]):
                v2 = tri_pairs[i, 2]
                v3 = tri_pairs[i, 3]
                vert[i][0] = v2
                vert[i][1] = v3
    kernel(vert,old_vert,tri_pairs)
    return vert