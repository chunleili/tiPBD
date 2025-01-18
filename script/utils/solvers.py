import numpy as np
import scipy
from time import perf_counter
import pyamg
from .construct_ml_manually import construct_ml_manually_3levels


def injectionP(A,b,x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "injectionP"
    print(f"Calculating {label}...")
    ml19 = pyamg.ruge_stuben_solver(A, max_coarse=400, keep=True, interpolation='injection')
    r = []
    _ = ml19.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})
    print("len(level)=", len(ml19.levels))


def GS(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "GS"
    print(f"Calculating {label}...")
    x4 = x0.copy()
    r = []
    for _ in range(maxiter+1):
        r.append(np.linalg.norm(b - A @ x4))
        pyamg.relaxation.relaxation.gauss_seidel(A=A, x=x4, b=b, iterations=1)
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def commonP(A1, b, x0, allres, P0, P1 , tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "commonP"
    print(f"Calculating {label}...")
    tt1 = perf_counter()
    ml18 = construct_ml_manually_3levels(A1,P0,P1)
    print("setup phase of commonP time=", perf_counter()-tt1)
    r = []
    tt = perf_counter()
    _ = ml18.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter, accel='cg')
    print("solve phase of common P time=", perf_counter()-tt)
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def SA_CG(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "SA_CG"
    print(f"Calculating {label}...")
    tt = perf_counter()
    ml17 = pyamg.smoothed_aggregation_solver(A, max_coarse=400, keep=True)
    print("setup phase of SA time=", perf_counter()-tt)
    r = []
    tt = perf_counter()
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    print("solve phase of SA time=", perf_counter()-tt)
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})
    print("len(level)=", len(ml17.levels))
    return ml17

def UA_CG(A, b, x0, allres, tol=1e-6, maxiter=100):
    label = "UA_CG"
    print(f"Calculating {label}...")
    tic1 = perf_counter()
    ml17 = pyamg.smoothed_aggregation_solver(A, smooth=None, max_coarse=400)
    toc1 = perf_counter()
    print("UA_CG Setup Time:", toc1-tic1)
    r = []
    tic2 = perf_counter()
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    toc2 = perf_counter()
    print("UA_CG Solve Time:", toc2-tic2)
    allres.append({"label":label, "r":r, "t":toc2-tic1})
    print("len(level)=", len(ml17.levels))
    print("iterations:", len(r)-1)



def UA_CG_chebyshev(A, b, x0, allres, tol=1e-6, maxiter=100):
    label = "UA_CG_chebyshev"
    print(f"Calculating {label}...")
    tic1 = perf_counter()
    ml17 = pyamg.smoothed_aggregation_solver(A, smooth=None, max_coarse=400, presmoother=('chebyshev', {'degree': 3}), postsmoother=('chebyshev', {'degree': 3}))
    toc1 = perf_counter()
    print("UA_CG_chebyshev Setup Time:", toc1-tic1)
    r = []
    tic2 = perf_counter()
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    toc2 = perf_counter()
    print("UA_CG_chebyshev Solve Time:", toc2-tic2)
    allres.append({"label":label, "r":r, "t":toc2-tic1})
    print("len(level)=", len(ml17.levels))


def UA_CG_jacobi(A, b, x0, allres, tol=1e-6, maxiter=100):
    label = "UA_CG_jacobi"
    print(f"Calculating {label}...")
    tic1 = perf_counter()
    ml17 = pyamg.smoothed_aggregation_solver(A, smooth=None, max_coarse=400, presmoother=('jacobi', {'omega': 1.0}), postsmoother=('jacobi', {'omega': 1.0}))
    toc1 = perf_counter()
    print("UA_CG_jacobi Setup Time:", toc1-tic1)
    r = []
    tic2 = perf_counter()
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    toc2 = perf_counter()
    print("UA_CG_jacobi Solve Time:", toc2-tic2)
    allres.append({"label":label, "r":r, "t":toc2-tic1})
    print("len(level)=", len(ml17.levels))

def UA_CG_GS(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "UA_CG"
    print(f"Calculating {label}...")
    ml17 = pyamg.smoothed_aggregation_solver(A, smooth=None, max_coarse=400, coarse_solver='gauss_seidel')
    r = []
    _ = ml17.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    print("len(level)=", len(ml17.levels))
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})

def CG(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "CG"
    print(f"Calculating {label}...")
    x6 = x0.copy()
    r = []
    r.append(np.linalg.norm(b - A @ x6))
    x6 = scipy.sparse.linalg.cg(A, b, x0=x0.copy(), tol=tol, maxiter=maxiter, callback=lambda x: r.append(np.linalg.norm(b - A @ x)))
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def diagCG(A,b,x0,allres,tol=1e-6,maxiter=100):
    label = "diagPCG"
    tic = perf_counter()
    print(f"Calculating {label}...")
    M = scipy.sparse.diags(1.0/A.diagonal())
    x7 = x0.copy()
    r = []
    r.append(np.linalg.norm(b - A @ x7))
    x7 = scipy.sparse.linalg.cg(A, b, x0=x0.copy(), rtol=tol, maxiter=maxiter, callback=lambda x: r.append(np.linalg.norm(b - A @ x)), M=M)
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def CAMG(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "CAMG" # classical AMG
    print(f"Calculating {label}...")
    ml1 = pyamg.ruge_stuben_solver(A)
    r = []
    _ = ml1.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter)
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def CAMG_CG(A,b,x0,allres,tol=1e-6,maxiter=100):
    tic = perf_counter()
    label = "CAMG_CG"
    print(f"Calculating {label}...")
    ml16 = pyamg.ruge_stuben_solver(A, max_coarse=400)
    r = []
    _ = ml16.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter, accel='cg')
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def SA(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "SA"
    print(f"Calculating {label}...")
    ml2 = pyamg.smoothed_aggregation_solver(A)
    r = []
    _ = ml2.solve(b, x0=x0.copy(), tol=tol, residuals=r, maxiter=maxiter)
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})

def adaptive_SA_CG(A,b,x0,allres,tol=1e-6,maxiter=100):
    tic = perf_counter()
    label = "adaptive SA+CG"
    print(f"Calculating {label}...")
    r = []
    ml = pyamg.aggregation.adaptive_sa_solver(A.astype(np.float64), max_coarse=400,  num_candidates=6)[0]
    _ = ml.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})

def adaptive_SA_CG_my(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "adaptive SA+CG(my)"
    print(f"Calculating {label}...")
    from script.amg_cuda_easy import amg_cuda_easy
    x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter, build_P_method="adaptive_SA")
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})

def nullspace_UA_CG(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "near kernel UA+CG"
    print(f"Calculating {label}...")
    from script.amg_cuda_easy import amg_cuda_easy
    x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter, build_P_method="nullspace")
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})


def SA_from_diagnostic(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "SA_from_diagnostic"
    B = np.ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()
    r = []
    ml = pyamg.smoothed_aggregation_solver(A,B=B,BH=BH, 
        strength=('symmetric', {'theta': 0.0}),
        smooth="jacobi",
        improve_candidates=None,
        aggregate="standard",
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        max_levels=15,
        max_coarse=300,
        coarse_solver="pinv")
    x = ml.solve(b, x0=x0, tol=tol, residuals=r, accel="cg", maxiter=maxiter, cycle="W")
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})

def SA_CG(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "SA+CG"
    print(f"Calculating {label}...")
    ml13 = pyamg.smoothed_aggregation_solver(A)
    r = []
    _ = ml13.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})

def nullspace_UA_CG(A, b, x0, allres, tol=1e-6, maxiter=100):
    tic = perf_counter()
    label = "nullspace UA+CG"
    print(f"Calculating {label}...")
    from script.amg_cuda_easy import amg_cuda_easy
    x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter, build_P_method="nullspace")
    toc = perf_counter()
    allres.append({"label":label, "r":r, "t":toc-tic})
    # label = "nullspace UA+CG"
    # print(f"Calculating {label}...")
    # # from script.amg_cuda_easy import amg_cuda_easy
    # # x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter, build_P_method="nullspace")
    # r = []
    # from engine.solver.build_Ps import calc_near_nullspace_GS
    # B = calc_near_nullspace_GS(A)
    # ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, symmetry='symmetric', B=B)
    # _ = ml.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
    # allres.append({"label":label, "r":r, "t":toc-tic})






# # GS
# label = "Gauss Seidel"
# print(f"Calculating {label}...")
# x4 = x0.copy()
# r = []
# for _ in range(maxiter*8+1):
#     r.append(np.linalg.norm(b - A @ x4))
#     pyamg.relaxation.relaxation.gauss_seidel(A=A, x=x4, b=b, iterations=1)
# allres.append({"label":label, "r":r, "t":toc-tic})



# # SA with strength algebraic_distance_epsilon3
# label = "SA+CG+Algebraic3.0"
# print(f"Calculating {label}...")
# ml8 = pyamg.smoothed_aggregation_solver(A, max_coarse=300, max_levels=15, strength=('algebraic_distance', {'epsilon': 3.0}))
# r = []
# _ = ml8.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
# allres.append({"label":label, "r":r, "t":toc-tic})


# # SA with strength affinity_4.0
# label = "SA+CG+Affinity4.0"
# print(f"Calculating {label}...")
# ml9 = pyamg.smoothed_aggregation_solver(A, max_coarse=300, max_levels=15, strength=('affinity', {'epsilon': 4.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
# r = []
# _ = ml9.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
# allres.append({"label":label, "r":r, "t":toc-tic})


# # blackbox
# label = "Blackbox"
# print(f"Calculating {label}...")
# r=[]
# x = pyamg.solve(A, b, x0, tol=tol, verb=False, residuals=r, maxiter=maxiter)
# conv10 = calc_conv(r)
# allres.append({"label":label, "r":r, "t":toc-tic})

# # rootnode
# label = "Rootnode+CG"
# print(f"Calculating {label}...")
# ml12 = pyamg.rootnode_solver(A)
# r = []
# x12 = ml12.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
# allres.append({"label":label, "r":r, "t":toc-tic})


# # SA+CG smooth='energy'
# label = "SA+CG smooth=energy"
# print(f"Calculating {label}...")
# ml14 = pyamg.smoothed_aggregation_solver(A, smooth='energy')
# r = []
# _ = ml14.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
# allres.append({"label":label, "r":r, "t":toc-tic})


# # label = "UA+CG coarse=GS"
# # print(f"Calculating {label}...")
# # ml18 = pyamg.smoothed_aggregation_solver(A, smooth=None, coarse_solver='gauss_seidel', max_coarse=300)
# # r = []
# # _ = ml18.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
# # allres.append({"label":label, "r":r, "t":toc-tic})

# label = "adaptive SA+CG"
# print(f"Calculating {label}...")
# r = []
# ml = pyamg.aggregation.adaptive_sa_solver(A.astype(np.float64), max_coarse=400,  num_candidates=6)[0]
# _ = ml.solve(b, x0=x0.copy(), tol=tol, residuals=r,maxiter=maxiter, accel='cg')
# allres.append({"label":label, "r":r, "t":toc-tic})

# label = "adaptive SA+CG(my)"
# print(f"Calculating {label}...")
# from script.amg_cuda_easy import amg_cuda_easy
# x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter, build_P_method="adaptive_SA")
# allres.append({"label":label, "r":r, "t":toc-tic})



def amg_cuda_solvers(A, b, x0, allres, tol=1e-6, maxiter=100, build_P_method="UA", smoother_type="jacobi", label=None):
    if label is None:
        label = f"{build_P_method}"
    print(f"Calculating {label}...")
    tic = perf_counter()
    from script.amg_cuda_easy import amg_cuda_easy
    x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter, build_P_method=build_P_method, smoother_type=smoother_type)
    toc = perf_counter()
    allres.append({"label":label, "r":r.tolist(), "t":toc-tic})

    
def amg_cuda_PCG(A, b, x0, allres, tol=1e-6, maxiter=100):
    label = f"PCG"
    print(f"Calculating {label}...")
    tic = perf_counter()
    from script.amg_cuda_easy import amg_cuda_easy
    x, r = amg_cuda_easy(A, b,  tol=tol, maxiter=maxiter,build_P_method="PCG")
    toc = perf_counter()
    allres.append({"label":label, "r":r.tolist(), "t":toc-tic})