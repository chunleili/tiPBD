
# ---------------------------------------------------------------------------- #
#                                soft                               #
# ---------------------------------------------------------------------------- #
@ti.func
def is_in_tet_func(p, p0, p1, p2, p3):
    A = ti.math.mat3([p1 - p0, p2 - p0, p3 - p0]).transpose()
    b = p - p0
    x = ti.math.inverse(A) @ b
    return ((x[0] >= 0 and x[1] >= 0 and x[2] >= 0) and x[0] + x[1] + x[2] <= 1), x


@ti.func
def tet_centroid_func(tet_indices, pos, t):
    a, b, c, d = tet_indices[t]
    p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
    p = (p0 + p1 + p2 + p3) / 4
    return p


@ti.kernel
def compute_all_centroid(pos: ti.template(), tet_indices: ti.template(), res: ti.template()):
    for t in range(tet_indices.shape[0]):
        a, b, c, d = tet_indices[t]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        p = (p0 + p1 + p2 + p3) / 4
        res[t] = p


@ti.kernel
def compute_R_kernel_new(
    fine_pos: ti.template(),
    fine_tet_indices: ti.template(),
    fine_centroid: ti.template(),
    coarse_pos: ti.template(),
    coarse_tet_indices: ti.template(),
    coarse_centroid: ti.template(),
    R: ti.types.sparse_matrix_builder(),
):
    for i in fine_centroid:
        p = fine_centroid[i]
        flag = False
        for tc in range(coarse_tet_indices.shape[0]):
            a, b, c, d = coarse_tet_indices[tc]
            p0, p1, p2, p3 = coarse_pos[a], coarse_pos[b], coarse_pos[c], coarse_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
            if flag:
                R[tc, i] += 1
                break
        if not flag:
            print("Warning: fine tet centroid {i} not in any coarse tet")


@ti.kernel
def compute_R_kernel_np(
    fine_pos: ti.template(),
    fine_tet_indices: ti.template(),
    fine_centroid: ti.template(),
    coarse_pos: ti.template(),
    coarse_tet_indices: ti.template(),
    coarse_centroid: ti.template(),
    R: ti.types.ndarray(),
):
    for i in fine_centroid:
        p = fine_centroid[i]
        flag = False
        for tc in range(coarse_tet_indices.shape[0]):
            a, b, c, d = coarse_tet_indices[tc]
            p0, p1, p2, p3 = coarse_pos[a], coarse_pos[b], coarse_pos[c], coarse_pos[d]
            flag, x = is_in_tet_func(p, p0, p1, p2, p3)
            if flag:
                R[tc, i] = 1
                break
        if not flag:
            print("Warning: fine tet centroid {i} not in any coarse tet")


@ti.kernel
def compute_R_based_on_kmeans_label(
    labels: ti.types.ndarray(dtype=int),
    R: ti.types.ndarray(),
):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if labels[j] == i:
                R[i, j] = 1


def compute_R_and_P(coarse, fine):
    # 计算所有四面体的质心
    print(">>Computing all tet centroid...")
    compute_all_centroid(fine.pos, fine.tet_indices, fine.tet_centroid)
    compute_all_centroid(coarse.pos, coarse.tet_indices, coarse.tet_centroid)

    # 计算R 和 P
    print(">>Computing P and R...")
    t = perf_counter()
    M, N = coarse.tet_indices.shape[0], fine.tet_indices.shape[0]
    R = np.zeros((M, N))
    compute_R_kernel_np(
        fine.pos, fine.tet_indices, fine.tet_centroid, coarse.pos, coarse.tet_indices, coarse.tet_centroid, R
    )
    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()
    print(f"Computing P and R done, time = {perf_counter() - t}")
    # print(f"writing P and R...")
    # R.mmwrite("R.mtx")
    # P.mmwrite("P.mtx")
    return R, P


def compute_R_and_P_kmeans(ist):
    print(">>Computing P and R...")
    t = perf_counter()

    from scipy.cluster.vq import vq, kmeans, whiten

    # 计算所有四面体的质心
    print(">>Computing all tet centroid...")
    compute_all_centroid(ist.pos, ist.tet_indices, ist.tet_centroid)

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = ist.tet_centroid.to_numpy()

    np.savetxt("tet_centroid.txt", input)

    N = input.shape[0]
    k = int(N / 100)
    print("N: ", N)
    print("k: ", k)

    # run kmeans
    input = whiten(input)
    print("whiten done")

    print("computing kmeans...")
    kmeans_centroids, distortion = kmeans(obs=input, k_or_guess=k, iter=20)
    labels, _ = vq(input, kmeans_centroids)

    print("distortion: ", distortion)
    print("kmeans done")

    # ----------------------------------- R and P --------------------------------- #
    # 计算R 和 P
    R = np.zeros((k, N), dtype=np.float32)

    compute_R_based_on_kmeans_label(labels, R)

    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()
    print(f"Computing P and R done, time = {perf_counter() - t}")

    print(f"writing P and R...")
    scipy.io.mmwrite("R.mtx", R)
    scipy.io.mmwrite("P.mtx", P)
    print(f"writing P and R done")

    return R, P




# ---------------------------------------------------------------------------- #
#                                cloth                               #
# ---------------------------------------------------------------------------- #
@ti.kernel
def compute_R_based_on_kmeans_label_triplets(
    labels: ti.types.ndarray(dtype=int),
    ii: ti.types.ndarray(dtype=int),
    jj: ti.types.ndarray(dtype=int),
    vv: ti.types.ndarray(dtype=int),
    new_M: ti.i32,
    NCONS: ti.i32
):
    cnt=0
    ti.loop_config(serialize=True)
    for i in range(new_M):
        for j in range(NCONS):
            if labels[j] == i:
                ii[cnt],jj[cnt],vv[cnt] = i,j,1
                cnt+=1



def compute_R_and_P_kmeans():
    print(">>Computing P and R...")
    t = time.perf_counter()

    from scipy.cluster.vq import vq, kmeans, whiten

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = edge_center.to_numpy()

    NCONS = NE
    global new_M
    print("NCONS: ", NCONS, "  new_M: ", new_M)

    # run kmeans
    input = whiten(input)
    print("whiten done")

    print("computing kmeans...")
    kmeans_centroids, distortion = kmeans(obs=input, k_or_guess=new_M, iter=5)
    labels, _ = vq(input, kmeans_centroids)

    print("distortion: ", distortion)
    print("kmeans done")

    # ----------------------------------- R and P --------------------------------- #
    # 将labels转换为R
    i_arr = np.zeros((NCONS), dtype=np.int32)
    j_arr = np.zeros((NCONS), dtype=np.int32)
    v_arr = np.zeros((NCONS), dtype=np.int32)
    compute_R_based_on_kmeans_label_triplets(labels, i_arr, j_arr, v_arr, new_M, NCONS)

    R = scipy.sparse.coo_array((v_arr, (i_arr, j_arr)), shape=(new_M, NCONS)).tocsr()
    P = R.transpose()
    print(f"Computing P and R done, time = {time.perf_counter() - t}")

    # print(f"writing P and R...")
    # scipy.io.mmwrite("R.mtx", R)
    # scipy.io.mmwrite("P.mtx", P)
    # print(f"writing P and R done")

    return R, P, labels, new_M