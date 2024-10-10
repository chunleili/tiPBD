

# taichi version: get the adjacent element shared vertices
def init_adj_share_v_ti(adj, nadj, ele):
    nele = ele.shape[0]
    max_nadj = max(nadj)
    print("max number of shared elements: ", max_nadj)
    # 共享顶点编号， 用法:shared_v[i,j,:]表示第i个ele的第j个adj ele共享的顶点
    shared_v = np.ones((nele, max_nadj, 3), dtype=np.int32) * (-1)
    # 共享顶点的个数， 用法：n_shared_v[i,j]表示第i个ele的j个adj ele共享的顶点个数
    n_shared_v = np.zeros((nele, max_nadj), dtype=np.int32)
    # 共享顶点在当前ele中是四面体的第几个(0-3)顶点, 用法：order_shared_v_in_cur[i,j,:] 表示第i个ele的第j个adj ele共享的顶点在当前ele中是第几个顶点。
    shared_v_order_in_cur = np.ones((nele, max_nadj, 3), dtype=np.int8) * (-1)
    # 共享顶点在邻接ele中是四面体的第几个(0-3)顶点, 用法：order_shared_v_in_adj[i,j,:] 表示第i个ele的第j个adj ele共享的顶点在邻接ele中是第几个顶点。
    shared_v_order_in_adj = np.ones((nele, max_nadj, 3), dtype=np.int8) * (-1)

    # 求两个长度为4的数组的交集
    @ti.func
    def intersect(a, b):   
        # a,b: 4个顶点的id, e:当前ele的id
        k=0 # 第几个共享的顶点， 0, 1, 2, 3
        c = ti.Vector([-1,-1,-1])         # 共享的顶点id存在c中
        order = ti.Vector([-1,-1,-1])     # 共享的顶点是当前ele的第几个顶点
        order2 = ti.Vector([-1,-1,-1])    # 共享的顶点是邻接ele的第几个顶点
        for i in ti.static(range(4)):     # i:当前ele的第i个顶点
            for j in ti.static(range(4)): # j:邻接ele的第j个顶点
                if a[i] == b[j]:
                    c[k] = a[i]         
                    order[k] = i          
                    order2[k] = j
                    k += 1
        return k, c, order, order2

    @ti.kernel
    def init_adj_share_v_kernel(adj:ti.types.ndarray(), 
                                nadj:ti.types.ndarray(), 
                                ele:ti.template(),
                                n_shared_v:ti.types.ndarray(), 
                                shared_v:ti.types.ndarray(),
                                shared_v_order_in_cur:ti.types.ndarray(), 
                                shared_v_order_in_adj:ti.types.ndarray()
                                ):
        nele = ele.shape[0]
        for i in range(nele):
            for j in range(nadj[i]):
                adj_id = adj[i,j]
                n, sharedv, order, order2 = intersect(ele[i], ele[adj_id])
                n_shared_v[i,j] = n
                for k in range(n):
                    shared_v[i, j, k] = sharedv[k]
                    shared_v_order_in_cur[i, j, k] = order[k]
                    shared_v_order_in_adj[i, j, k] = order2[k]


    init_adj_share_v_kernel(adj, nadj, ele, n_shared_v, shared_v, shared_v_order_in_cur, shared_v_order_in_adj)

    return n_shared_v, shared_v, shared_v_order_in_cur, shared_v_order_in_adj


if __name__ == "__main__":
    n_shared_v, shared_v, shared_v_order_in_cur, shared_v_order_in_adj = init_adj_share_v_ti(adjacent, num_adjacent, ist.tet_indices)
