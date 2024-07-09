import ctypes
import numpy as np
import os
from time import perf_counter

g_vcycle = None
g_vcycle_cached_levels = None
vcycle_has_course_solve = False

def init_g_vcycle(levels, chebyshev_coeff=None):
    global g_vcycle
    global g_vcycle_cached_levels

    if g_vcycle is None:
        os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin")
        g_vcycle = ctypes.cdll.LoadLibrary('./cpp/mgcg_cuda/lib/fast-vcycle-gpu.dll')
        
        
        g_vcycle.fastmg_copy_outer2init_x.argtypes = []
        g_vcycle.fastmg_set_outer_x.argtypes = [ctypes.c_size_t] * 2
        g_vcycle.fastmg_set_outer_b.argtypes = [ctypes.c_size_t] * 2
        g_vcycle.fastmg_init_cg_iter0.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_init_cg_iter0.restype = ctypes.c_float
        g_vcycle.fastmg_do_cg_itern.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        g_vcycle.fastmg_fetch_cg_final_x.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_setup.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_set_coeff.argtypes = [ctypes.c_size_t] * 2
        g_vcycle.fastmg_set_init_x.argtypes = [ctypes.c_size_t] * 2
        g_vcycle.fastmg_set_init_b.argtypes = [ctypes.c_size_t] * 2
        g_vcycle.fastmg_get_coarsist_size.argtypes = []
        g_vcycle.fastmg_get_coarsist_size.restype = ctypes.c_size_t
        g_vcycle.fastmg_get_coarsist_b.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_set_coarsist_x.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_get_finest_x.argtypes = [ctypes.c_size_t]
        g_vcycle.fastmg_set_lv_csrmat.argtypes = [ctypes.c_size_t] * 11
        g_vcycle.fastmg_vcycle_down.argtypes = []
        g_vcycle.fastmg_coarse_solve.argtypes = []
        g_vcycle.fastmg_vcycle_up.argtypes = []

    if g_vcycle_cached_levels != id(levels):
        print('Setup detected! reuploading A, R, P matrices')
        g_vcycle_cached_levels = id(levels)
        assert chebyshev_coeff is not None
        coeff_contig = np.ascontiguousarray(chebyshev_coeff, dtype=np.float32)
        g_vcycle.fastmg_setup(len(levels))
        g_vcycle.fastmg_set_coeff(coeff_contig.ctypes.data, coeff_contig.shape[0])
        for lv in range(len(levels)):
            for which, matname in zip([1, 2, 3], ['A', 'R', 'P']):
                mat = getattr(levels[lv], matname)
                if mat is not None:
                    data_contig = np.ascontiguousarray(mat.data, dtype=np.float32)
                    indices_contig = np.ascontiguousarray(mat.indices, dtype=np.int32)
                    indptr_contig = np.ascontiguousarray(mat.indptr, dtype=np.int32)
                    # print(data_contig)
                    # print(indices_contig)
                    # if matname == 'A':
                        # print('UUUO', lv, indices_contig[0], indices_contig[-1], indices_contig.shape)
                    # print(indptr_contig)
                    g_vcycle.fastmg_set_lv_csrmat(lv, which, data_contig.ctypes.data, data_contig.shape[0],
                                                  indices_contig.ctypes.data, indices_contig.shape[0],
                                                  indptr_contig.ctypes.data, indptr_contig.shape[0],
                                                  mat.shape[0], mat.shape[1], mat.nnz)

def new_V_cycle(levels):
    assert g_vcycle
    g_vcycle.fastmg_vcycle_down()
    if vcycle_has_course_solve:
        g_vcycle.fastmg_coarse_solve()
    else:
        coarsist_size = g_vcycle.fastmg_get_coarsist_size()
        coarsist_b_empty = np.empty(shape=(coarsist_size,), dtype=np.float32)
        coarsist_b = np.ascontiguousarray(coarsist_b_empty, dtype=np.float32)
        g_vcycle.fastmg_get_coarsist_b(coarsist_b.ctypes.data)
        # ##################33
        # np.save(f'/tmp/new_b_{frame}-{bcnt}.npy', coarsist_b)
        # ##################33
        def coarse_solver(A, b):
            res = np.linalg.solve(A.toarray(), b)
            return res
        coarsist_x = coarse_solver(levels[len(levels) - 1].A, coarsist_b)
        coarsist_x_contig = np.ascontiguousarray(coarsist_x, dtype=np.float32)
        g_vcycle.fastmg_set_coarsist_x(coarsist_x_contig.ctypes.data)
    g_vcycle.fastmg_vcycle_up()


def new_amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    init_g_vcycle(levels)
    assert g_vcycle

    assert x0 is not None
    tic_amgcg = perf_counter()
    x0_contig = np.ascontiguousarray(x0, dtype=np.float32)
    g_vcycle.fastmg_set_outer_x(x0_contig.ctypes.data, x0_contig.shape[0])
    t_vcycle = 0.0
    b_contig = np.ascontiguousarray(b, dtype=np.float32)
    g_vcycle.fastmg_set_outer_b(b_contig.ctypes.data, b.shape[0])
    residuals_empty = np.empty(shape=(maxiter+1,), dtype=np.float32)
    residuals = np.ascontiguousarray(residuals_empty, dtype=np.float32)
    bnrm2 = g_vcycle.fastmg_init_cg_iter0(residuals.ctypes.data) # init_b = r = b - A@x; residuals[0] = normr
    atol = bnrm2 * tol
    iteration = 0
    for iteration in range(maxiter):
        if residuals[iteration] < atol:
            break
        tic_vcycle = perf_counter()
        g_vcycle.fastmg_copy_outer2init_x()
        new_V_cycle(levels)
        toc_vcycle = perf_counter()
        t_vcycle += toc_vcycle - tic_vcycle
        # print(f"Once V_cycle time: {toc_vcycle - tic_vcycle:.4f}s")
        g_vcycle.fastmg_do_cg_itern(residuals.ctypes.data, iteration)
    x_empty = np.empty_like(x0, dtype=np.float32)
    x = np.ascontiguousarray(x_empty, dtype=np.float32)
    g_vcycle.fastmg_fetch_cg_final_x(x.ctypes.data)
    residuals = residuals[:iteration+1]
    toc_amgcg = perf_counter()
    t_amgcg = toc_amgcg - tic_amgcg
    print(f"Total V_cycle time in one amg_cg_solve: {t_vcycle:.4f}s")
    print(f"Total time of amg_cg_solve: {t_amgcg:.4f}s")
    print(f"Time of CG(exclude v-cycle): {t_amgcg - t_vcycle:.4f}s")
    return (x),  residuals  

