import numpy as np
import pyamg
import logging
from time import perf_counter
import ctypes

def build_Ps(A,args,extlib=None, verbose=False):
    """Build a list of prolongation matrices Ps from A """
    method = args.build_P_method
    logging.info(f"build P by method:{method}")
    tic = perf_counter()
    if method == 'UA':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None, symmetry='symmetric')
    elif method == 'SA' :
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400,symmetry='symmetric')
    elif method == 'CAMG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400)
    elif method == 'adaptive_SA':
        ml = pyamg.aggregation.adaptive_sa_solver(A.astype(np.float64), max_coarse=400,  num_candidates=6)[0]
    elif method == 'nullspace':
        B = calc_near_nullspace_GS(A)
        logging.info(f"B shape: {B.shape}")
        logging.info(f"B: {B}")
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, symmetry='symmetric', B=B)
    elif method == 'nullspace_amg':
        B = calc_near_nullspace_amg(A)
        logging.info(f"B shape: {B.shape}")
        logging.info(f"B: {B}")
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, symmetry='symmetric', B=B)
    elif method == 'rbm':
        B_in = np.load("rbm.npy")
        B = B_in
        logging.info(f"B shape: {B.shape}")
        logging.info(f"B: {B}")
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, symmetry='symmetric', B=B)
    elif method == 'algebraic3.0':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('algebraic_distance', {'epsilon': 3.0}))
    elif method == 'affinity4.0':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('affinity', {'epsilon': 4.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    elif method == 'strength0.1':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.1 }))    
    elif method == 'strength0.2':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.2 }))    
    elif method == 'strength0.25':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.25 }))
    elif method == 'strength0.3':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.3 }))
    elif method == 'strength0.4':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.4 }))
    elif method == 'strength0.5':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.5 }))
    elif method == 'evolution':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('evolution', {'k': 2, 'proj_type': 'l2', 'epsilon': 4.0}))
    elif method == 'improve_candidate':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth = None, improve_candidates=(('block_gauss_seidel',{'sweep': 'symmetric','iterations': 4}),None), symmetry='symmetric', strength=('symmetric',{'theta' : 0.1 }))
    elif method == 'strength_energy':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('energy_based',{'theta' : 0.25 })) 
    elif method == 'strength_classical':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('classical')) 
    elif method == 'strength_distance':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('distance')) 
    elif method == 'aggregate_standard':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='standard')
    elif method == 'aggregate_naive':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='naive')
    elif method == 'aggregate_lloyd':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='lloyd')
    elif method == 'aggregate_pairwise':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', aggregate='pairwise')
    elif method == 'diagonal_dominance':
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=400, smooth=None,symmetry='symmetric', strength=('symmetric',{'theta' : 0.1 }),diagonal_dominance=True)
    else:
        raise ValueError(f"Method {method} not recognized")

    num_levels = len(ml.levels)

    if args.use_cuda and extlib is not None:
        extlib.fastmg_setup_nl.argtypes = [ctypes.c_size_t]
        extlib.fastmg_setup_nl(num_levels)
    
    # if(verbose):
    logging.info(ml)

    Ps = []
    for i in range(len(ml.levels)-1):
        P = ml.levels[i].P.tocsr()
        if args.filter_P=="fileter":
            P = do_filter_P(P,0.25)
        elif args.filter_P=="01":
            P = do_set_01_P(P)
        elif args.filter_P=="avg":
            P = do_set_avg_P(P)
        Ps.append(P)

        if args.scale_RAP and args.use_cuda:
            # scale RAP by avg size of aggregates
            # get scale from nnz of each column of P
            s = calc_RAP_scale(P)
            extlib.fastmg_scale_RAP(s, i)

    toc = perf_counter()
    logging.info(f"Build P Time:{toc-tic:.2f}s")

    # logger2.info(f"logger2 {method} {toc-tic}")
    file = args.out_dir+'/build_P_time.log'
    with open(file, 'a') as f:
        f.write(f"{method} {toc-tic}\n")
    return Ps




def calc_near_nullspace_GS(A):
    n=6
    print("Calculating near nullspace")
    tic = perf_counter()
    B = np.zeros((A.shape[0],n), dtype=np.float64)
    # from pyamg.relaxation.relaxation import gauss_seidel
    from engine.solver.iterative_solver import GaussSeidelSolver
    def get_A0():
        return A
    gs = GaussSeidelSolver(get_A0)
    Amax = np.max(np.abs(A.data))
    print(f"Amax: {Amax}")
    x0_rand = np.random.rand(A.shape[0],6) * Amax
    for i in range(n):
        b = np.zeros(A.shape[0]) 
        x,_ = gs.run(b, x0_rand[:,i], maxiter=30)
        x = (1.0/np.max(np.abs(x))) * x
        # x = (1.0/np.linalg.norm(x)) * x
        B[:,i] = x
        print(f"norm B {i}: {np.linalg.norm(B[:,i])}")
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B


def calc_near_nullspace_amg(A):
    n=6
    print("Calculating near nullspace")
    tic = perf_counter()
    B = np.zeros((A.shape[0],n), dtype=np.float64)
    from engine.solver.amg_cuda_easy import amg_cuda_easy
    for i in range(n):
        x,_ = amg_cuda_easy(A, np.ones(A.shape[0]))
        x = (1.0/np.max(np.abs(x))) * x
        # x = (1.0/np.linalg.norm(x)) * x
        B[:,i] = x
        print(f"norm B {i}: {np.linalg.norm(B[:,i])}")
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B




def do_filter_P(P, theta=0.25):
    # filter out the small values in each column of P
    # small value: |val| < 0.25 |max_val|
    logging.info(f"Filtering P, shape: {P.shape}")
    P = P.tocsc()
    indices, indptr, data = P.indices, P.indptr, P.data
    for j in range(P.shape[1]):
        col_start = indptr[j]
        col_end = indptr[j + 1]
        col_data = data[col_start:col_end]
        max_val = np.abs(col_data).max()
        ...
        for i in range(col_start, col_end):
            if np.abs(data[i]) < theta * max_val:
                data[i] = 0
    P.eliminate_zeros()
    return P.tocsr()


def do_set_01_P(P):
    # for all non-zero values in P, set them to 1
    logging.info(f"set 01 P, shape: {P.shape}")
    P.data[:] = 1
    P = P.tocsr()
    logging.info(f"set 01 P done")
    return P


def do_set_avg_P(P):
    # for all non-zero values in P, set them to 1
    logging.info(f"set avg P, shape: {P.shape}")
    P.data[:] = 1
    # for each column, set the each value to avg, and sum to 1.0
    P = P.tocsc()
    for j in range(P.shape[1]):
        col_start = P.indptr[j]
        col_end = P.indptr[j + 1]
        col_data = P.data[col_start:col_end]
        col_sum = np.sum(col_data)
        if col_sum != 0:
            P.data[col_start:col_end] /= col_sum
    P = P.tocsr()
    logging.info(f"set avg P done")
    return P


def calc_RAP_scale(P):
    logging.info(f"get RAP scale from nnz of each column of P, shape: {P.shape}")
    P = P.tocsc()
    nnz_col = np.zeros(P.shape[1], dtype=np.int32)
    for j in range(P.shape[1]):
        col_start = P.indptr[j]
        col_end = P.indptr[j + 1]
        nnz_col[j] = col_end - col_start #size of aggregate
    scale = 1.0/nnz_col.mean()
    logging.info(f"RAP scale={scale}")
    return scale

