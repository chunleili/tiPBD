import pyamg
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from pyamg.krylov import cg
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial
from time import perf_counter
from scipy.linalg import pinv

smoother = 'gauss_seidel'
update_coarse_solver = False

from utils.build_Ps import build_Ps
from utils.build_levels import build_levels
from utils.smoothers import presmoother, postsmoother, chebyshev
from utils.coarse_solver import coarse_solver
from utils.amg_cg_solve import amg_cg_solve


def mesh_to_coo(readpath):
    import meshio
    mesh = meshio.read(readpath)
    p = mesh.points
    p = p.ravel(order='F')
    coo = p.reshape((p.size, 1))
    return coo


def calc_rigidbodymodes():
    # coo mesh grid point coordinates in [x0,x1,x2,...,xn,y0,y1,y2,...,yn,z0,z1,z2,...,zn]
    import sys
    import os
    from pathlib import Path
    sys.path.append(os.getcwd())
    prjdir = Path(__file__).resolve().parent.parent
    print(prjdir/'pybind/build/Release')
    sys.path.append(str(prjdir/'pybind/build/Release'))
    sys.path.append(str(prjdir/'pybind/build/Debug'))


    from utils.define_to_read_dir import prj_dir, case_name
    meshpath = prj_dir + f"result/{case_name}/mesh/0010.ply"
    coo = mesh_to_coo(meshpath)

    import rigid_body_modes # type: ignore
    import numpy as np
    B = np.zeros(shape=coo.shape)  
    transpose = False 
    ndim = 3
    B = rigid_body_modes.rigid_body_modes(ndim, coo, B, transpose)
    return B


def calc_near_nullspace_GS(A):
    n=6
    print("Calculating near nullspace")
    tic = perf_counter()
    B = np.zeros((A.shape[0],n), dtype=np.float64)
    from pyamg.relaxation.relaxation import gauss_seidel
    for i in range(n):
        x = np.ones(A.shape[0]) + 1e-2*np.random.rand(A.shape[0])
        b = np.zeros(A.shape[0]) 
        gauss_seidel(A,x,b,iterations=20, sweep='forward')
        B[:,i] = x
        print(f"norm B {i}: {np.linalg.norm(B[:,i])}")
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B


def calc_near_nullspace_MG(A):
    print("Calculating near nullspace")
    tic = perf_counter()
    b = np.zeros(A.shape[0])
    B = np.zeros((A.shape[0],6), dtype=np.float64)
    Ps = build_Ps(A, method='UA')
    levels = build_levels(A, Ps)
    for i in range(6):
        x0 = np.ones(A.shape[0]) + 1e-2*np.random.rand(A.shape[0])
        x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=50, tol=1e-3)
        B[:,i] = x
        print(f"norm B {i}: {np.linalg.norm(B[:,i])}")
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B


def calc_near_nullspace_CG(A):
    import scipy.sparse.linalg
    print("Calculating near nullspace")
    tic = perf_counter()
    b = np.zeros(A.shape[0]) + 1e-4*np.random.rand(A.shape[0])
    B = np.zeros((A.shape[0],6), dtype=np.float64)
    for i in range(6):
        x0 = np.ones(A.shape[0]) + 1e-2*np.random.rand(A.shape[0])
        x,_ = scipy.sparse.linalg.minres(A, b, x0=x0.copy(), tol=1e-4, maxiter=50)
        B[:,i] = x
        print(f"norm B {i}: {np.linalg.norm(B[:,i])}")
    toc = perf_counter()
    print("Calculating near nullspace Time:", toc-tic)
    return B


def main(postfix='F10-0'):
    import os, sys
    sys.path.append(os.getcwd())
    from utils.load_A_b import load_A_b
    from utils.define_to_read_dir import prj_dir, case_name
    from utils.solvers import UA_CG, UA_CG_chebyshev, UA_CG_jacobi, CG, diagCG
    from collections import namedtuple
    from utils.plot_residuals import plot_residuals_all
    from utils.postprocess_residual import print_allres_time, calc_conv, print_df_newnew
    from utils.parms import maxiter
    Residual = namedtuple('Residual', ['label','r', 't'])
    global smoother, chebyshev, levels


    A, b = load_A_b(postfix)
    x0 = np.zeros_like(b)

    allres = []

    # setup phase
    t0= perf_counter()
    B = calc_near_nullspace_GS(A)
    Ps = build_Ps(A, method='nullspace', B=B)
    levels = build_levels(A, Ps)
    t1 = perf_counter()
    print('Setup Time:', t1-t0)


    A, b = load_A_b(postfix='F20-0')
    levels = build_levels(A, Ps)


    # solve phase
    smoother = 'gauss_seidel'
    tic = perf_counter()
    x0 = np.zeros_like(b)
    x,residuals = amg_cg_solve(levels, b, x0=x0.copy(), maxiter=maxiter, tol=1e-6)
    toc = perf_counter()
    print("solve phase:", toc-tic)
    allres.append(Residual('nullspace', residuals, toc-tic))
    print("iterations:", len(residuals)-1)


    UA_CG(A,b,x0,allres)


    print_df_newnew(allres)
    plot_residuals_all(allres,use_markers=True)


if __name__ == "__main__":
    # for f in [10,20,30,40,50,60]:
    for f in [10]:
        postfix = f"F{f}-0"
        print(f"Postfix: {postfix}")
        main(postfix)
