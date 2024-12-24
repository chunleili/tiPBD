import taichi as ti
import numpy as np
from time import perf_counter
import os

# ---------------------------------------------------------------------------- #
#                               directly  fill A                               #
# ---------------------------------------------------------------------------- #
def init_adj_ele(eles):
    vertex_to_eles = {}
    for ele_index, (v1, v2, v3, v4) in enumerate(eles):
        if v1 not in vertex_to_eles:
            vertex_to_eles[v1] = set()
        if v2 not in vertex_to_eles:
            vertex_to_eles[v2] = set()
        if v3 not in vertex_to_eles:
            vertex_to_eles[v3] = set()
        if v4 not in vertex_to_eles:
            vertex_to_eles[v4] = set()

        vertex_to_eles[v1].add(ele_index)
        vertex_to_eles[v2].add(ele_index)
        vertex_to_eles[v3].add(ele_index)
        vertex_to_eles[v4].add(ele_index)

    # sort 
    for k in vertex_to_eles.keys():
        vertex_to_eles[k] = set(sorted(list(vertex_to_eles[k])))

    all_adjacent_eles = {}

    for ele_index in range(len(eles)):
        v1, v2, v3, v4 = eles[ele_index]
        adjacent_eles = vertex_to_eles[v1] | vertex_to_eles[v2] | vertex_to_eles[v3] | vertex_to_eles[v4]
        adjacent_eles.remove(ele_index)  # 移除本身
        all_adjacent_eles[ele_index] = sorted(list(adjacent_eles))# sort
    return all_adjacent_eles, vertex_to_eles


def init_adj_ele_ti(eles):
    eles = eles
    nele = eles.shape[0]
    v2e = ti.field(dtype=ti.i32, shape=(nele, 200))
    nv2e = ti.field(dtype=ti.i32, shape=nele)

    @ti.kernel
    def calc_vertex_to_eles_kernel(eles: ti.template(), v2e: ti.template(), nv2e: ti.template()):
        # v2e: vertex to element
        # nv2e: number of elements sharing the vertex
        for e in range(eles.shape[0]):
            v1, v2, v3, v4 = eles[e]
            for v in ti.static([v1, v2, v3, v4]):
                k = nv2e[v]
                v2e[v, k] = e
                nv2e[v] += 1

    calc_vertex_to_eles_kernel(eles, v2e, nv2e)
    # v2e = v2e.to_numpy()
    # nv2e = nv2e.to_numpy()

# transfer one-to-multiple map dict to ndarray
def dict_to_ndarr(d:dict)->np.ndarray:
    lengths = np.array([len(v) for v in d.values()])
    max_len = max(lengths)
    arr = np.ones((len(d), max_len), dtype=np.int32) * (-1)
    for i, (k, v) in enumerate(d.items()):
        arr[i, :len(v)] = v
    return arr, lengths


def init_A_CSR_pattern(num_adj, adj):
    nrows = len(num_adj)
    nonz = np.sum(num_adj)+nrows
    indptr = np.zeros(nrows+1, dtype=np.int32)
    indices = np.zeros(nonz, dtype=np.int32)
    data = np.zeros(nonz, dtype=np.float32)
    indptr[0] = 0
    for i in range(0,nrows):
        num_adj_i = num_adj[i]
        indptr[i+1]=indptr[i] + num_adj_i + 1
        indices[indptr[i]:indptr[i+1]-1]= adj[i][:num_adj_i]
        indices[indptr[i+1]-1]=i
    assert indptr[-1] == nonz
    return data, indices, indptr


def csr_index_to_coo_index(indptr, indices):
    ii, jj = np.zeros_like(indices), np.zeros_like(indices)
    nrows = len(indptr)-1
    for i in range(nrows):
        ii[indptr[i]:indptr[i+1]]=i
    jj[:]=indices[:]
    return ii, jj


def initFill_tocuda(ist, extlib):
    from ctypes import c_int, c_float
    import numpy.ctypeslib as ctl
    arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
    arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
    extlib.fastFillSoft_init_from_python_cache_lessmem.argtypes = [c_int]*2  + [arr_float] + [arr_int]*3 + [c_int]

    extlib.fastFillSoft_init_from_python_cache_lessmem(
            ist.NT,
            ist.MAX_ADJ,
            ist.spmat_data,
            ist.spmat_indices,
            ist.spmat_indptr,
            ist.ii,
            ist.nnz)
    extlib.fastFillSoft_set_data(ist.tet_indices.to_numpy(), ist.NT, ist.inv_mass.to_numpy(), ist.NV, ist.pos.to_numpy(), ist.alpha_tilde.to_numpy())


def mem_usage(ist):
    # 内存占用
    # 将字节转换为GB
    def bytes_to_gb(bytes):
        return bytes / (1024 ** 3)

    data_memory_gb = bytes_to_gb(ist.spmat_data.nbytes)
    indices_memory_gb = bytes_to_gb(ist.spmat_indices.nbytes)
    indptr_memory_gb = bytes_to_gb(ist.spmat_indptr.nbytes)
    ii_memory_gb = bytes_to_gb(ist.ii.nbytes)
    total_memory_gb = (data_memory_gb + indices_memory_gb + indptr_memory_gb + ii_memory_gb)

    # 打印每个数组的内存占用和总内存占用（GB）
    print(f"data memory: {data_memory_gb:.2f} GB")
    print(f"indices memory: {indices_memory_gb:.2f} GB")
    print(f"indptr memory: {indptr_memory_gb:.2f} GB")
    print(f"ii memory: {ii_memory_gb:.2f} GB")
    print(f"Total memory: {total_memory_gb:.2f} GB")


def init_direct_fill_A(ist, extlib=None):
    args = ist.args
    cache_file_name = f'cache_initFill_{os.path.basename(args.model_path)}.npz'
    if args.use_cache and os.path.exists(cache_file_name):
        tic = perf_counter()
        print(f"Found cache {cache_file_name}. Loading cached data...")
        npzfile = np.load(cache_file_name)
        ist.spmat_data = npzfile['data']
        ist.spmat_indices = npzfile['indices']
        ist.spmat_indptr = npzfile['indptr']
        ist.ii = npzfile['ii']
        ist.nnz = int(npzfile['nnz'])
        ist.jj = ist.spmat_indices # No need to save jj,  indices is the same as jj
        ist.MAX_ADJ = int(npzfile['MAX_ADJ'])
        print(f"MAX_ADJ: {ist.MAX_ADJ}")
        mem_usage(ist)
        if args.use_cuda and extlib is not None:
            initFill_tocuda(ist, extlib)
        print(f"Loading cache time: {perf_counter()-tic:.3f}s")
        return

    print(f"No cached data found, initializing...")

    tic1 = perf_counter()
    print("Initializing adjacent elements and abc...")
    adjacent, v2e = init_adj_ele(eles=ist.tet_indices.to_numpy())
    # adjacent = init_adj_ele_ti(eles=ist.tet_indices)
    num_adjacent = np.array([len(v) for v in adjacent.values()])
    AVG_ADJ = np.mean(num_adjacent)
    ist.MAX_ADJ = max(num_adjacent)
    print(f"MAX_ADJ: {ist.MAX_ADJ}")
    print(f"AVG_ADJ: {AVG_ADJ}")
    print(f"init_adjacent time: {perf_counter()-tic1:.3f}s")

    tic = perf_counter()
    ist.spmat_data, ist.spmat_indices, ist.spmat_indptr = init_A_CSR_pattern(num_adjacent, adjacent)
    ist.ii, ist.jj = csr_index_to_coo_index(ist.spmat_indptr, ist.spmat_indices)
    ist.nnz = len(ist.spmat_data)
    # nnz_each_row = num_adjacent[:] + 1
    print(f"init_A_CSR_pattern time: {perf_counter()-tic:.3f}s")
    
    tic = perf_counter()
    adjacent,_ = dict_to_ndarr(adjacent)
    print(f"dict_to_ndarr time: {perf_counter()-tic:.3f}s")

    tic = perf_counter()
    print(f"init_adj_share_v time: {perf_counter()-tic:.3f}s")
    print(f"initFill done")

    mem_usage(ist)

    if args.use_cache:
        print(f"Saving cache to {cache_file_name}...")
        np.savez(cache_file_name, data=ist.spmat_data, indices=ist.spmat_indices, indptr=ist.spmat_indptr, ii=ist.ii, nnz=ist.nnz, MAX_ADJ=ist.MAX_ADJ)
        print(f"{cache_file_name} saved")
    if args.use_cuda:
        initFill_tocuda(ist,extlib)
