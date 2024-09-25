import numpy as np

# edge: (num_edge, 2)
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


def init_bending():
    tri = np.array([
        [1,3,2],
        [0,1,2],
    ])

    tri_neighbor = find_tri_neighbors(tri)
    print("邻居边编号列表:", tri_neighbor)

    # tri_pairs有四个点，第四个点是另一个三角形的点
    tri_pairs = build_tri_pairs(tri, tri_neighbor)
    print("三角形对列表:", tri_pairs)

if __name__ == "__main__":
    init_bending()
    print("Done.")