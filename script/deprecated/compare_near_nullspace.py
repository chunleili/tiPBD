import pyamg
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import bicgstab
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from pyamg.relaxation.relaxation import polynomial
from time import perf_counter
from scipy.linalg import pinv
from scipy.io import mmread

smoother = 'gauss_seidel'
update_coarse_solver = False

from utils.build_Ps import build_Ps
from utils.build_levels import build_levels
from utils.smoothers import presmoother, postsmoother, chebyshev
from utils.coarse_solver import coarse_solver
from utils.amg_cg_solve import amg_cg_solve
from utils.calc_RBM import calc_RBM3d


def mesh_to_coo(readpath):
    import meshio
    mesh = meshio.read(readpath)
    p = mesh.points
    # p = p.ravel(order='F')
    p = p.ravel()
    coo = p.reshape((p.size, 1))
    return coo


def calc_rigidbodymodes():
    # coo mesh grid point coordinates in [x1,y1,z1,x2,y2,z2,...,xn,yn,zn] format
    import sys
    import os
    from pathlib import Path
    sys.path.append(os.getcwd())
    prjdir = Path(__file__).resolve().parent.parent
    print(prjdir/'pybind/build/Release')
    sys.path.append(str(prjdir/'pybind/build/Release'))
    sys.path.append(str(prjdir/'pybind/build/Debug'))


    from utils.define_to_read_dir import prj_dir, case_name
    meshpath = prj_dir + f"result/{case_name}/mesh/0000.ply"
    coo = mesh_to_coo(meshpath)

    import rigid_body_modes # type: ignore
    import numpy as np
    B = np.zeros(shape=coo.shape)  
    transpose = False 
    ndim = 3
    B = rigid_body_modes.rigid_body_modes(ndim, coo, B, transpose)
    return np.array(B)


def RBM_from_dx_to_dlam(G, invM, B):
    # dx = invM @ G.T @ dlam
    # we need to solve a linear system to get dlam
    # invM @ G.T @ dlam = dx_RBM
    # Ax=b, where A = invM @ G.T, b = dx_RBM
    # But for easier to solve, we need to make A square, so we multiply by G
    # A = G @ invM @ G.T , b = G @ dx_RBM
    n = B.shape[1]
    newB = []
    for i in range(n):
        dx_RBM = B[:,i]
        A_ = G @ invM @ G.T
        A_ = A_.astype(np.float64)
        b_ = G@dx_RBM 
        # use GS to solve Ax=b
        dlam = np.zeros_like(b_)
        gauss_seidel(A_, dlam, b_, iterations=20, sweep='forward')
        newB.append(dlam)
    return np.array(newB).T


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


def main(postfix='F0-0'):
    import os, sys
    sys.path.append(os.getcwd())
    from utils.load_A_b import load_A_b
    from utils.define_to_read_dir import to_read_dir
    from utils.define_to_read_dir import prj_dir, case_name
    from utils.solvers import UA_CG, UA_CG_chebyshev, UA_CG_jacobi, CG, diagCG
    from collections import namedtuple
    from utils.plot_residuals import plot_residuals_all
    from utils.postprocess_residual import print_allres_time, calc_conv, print_df_newnew
    from utils.parms import maxiter
    Residual = namedtuple('Residual', ['label','r', 't'])
    global smoother, chebyshev, levels
    
    # calc B by transfer back RBM to dlam
    # G = scipy.sparse.load_npz(to_read_dir+f"G_{postfix}.npz")
    # M_inv = scipy.sparse.load_npz(to_read_dir+f"Minv_{postfix}.npz")
    # B = calc_RBM3d()
    # B[:,:] += np.random.rand(B.shape[0], B.shape[1])*1e-4
    # B = RBM_from_dx_to_dlam(G, M_inv, B)

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
    main()
