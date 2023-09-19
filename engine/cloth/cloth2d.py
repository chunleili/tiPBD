"""
Then we use XPBD-FEM (gpu version, Jacobian solver) to simulate the deformation of 2D object
"""
import taichi as ti
import time
from time import perf_counter
from taichi.lang.ops import sqrt
import scipy
import numpy as np
from collections import namedtuple
import scipy.sparse as sparse

ti.init(arch=ti.gpu, kernel_profiler=True)

h = 0.003  # timestep size

compliance = 1.0e-3  # Fat Tissuse compliance, for more specific material,please see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h
                      )  # timestep related compliance, see XPBD paper
N = 10
NF = 2 * N**2  # number of faces
NV = (N + 1)**2  # number of vertices
pos = ti.Vector.field(2, float, NV)
oldPos = ti.Vector.field(2, float, NV)
vel = ti.Vector.field(2, float, NV)  # velocity of particles
invMass = ti.field(float, NV)  #inverse mass of particles
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # D_m^{-1}
F = ti.Matrix.field(2, 2, float, NF)  # deformation gradient
lagrangian = ti.field(float, NF)  # lagrangian multipliers
gravity = ti.Vector([0, -1.2])
MaxIte = 20
NumSteps = 1

gradient = ti.Vector.field(2, float, 3 * NF)
dLambda = ti.field(float, NF)
omega = 0.5
# For validation
dualResidual = ti.field(float, ())
primalResidual = ti.field(float, ())

attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


constraints = ti.field(float, NF)  # s
gradC = ti.Vector.field(2, float, (NF,3))  # gradient of constraints
pos_mid = ti.Vector.field(2, float, NV)  # mid pos
output_A_flag = False
centroids = ti.Vector.field(2, float, NF)  # centroids 


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) * 0.05 + ti.Vector([0.25, 0.25])
        oldPos[k] = pos[k]
        vel[k] = ti.Vector([0, 0])
        invMass[k] = 1.0
    for i in range(N + 1):
        k = i * (N + 1) + N
        invMass[k] = 0.0
    k0 = N
    k1 = (N + 2) * N
    invMass[k0] = 0.0
    invMass[k1] = 0.0
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


@ti.kernel
def resetLagrangian():
    for i in range(NF):
        lagrangian[i] = 0.0


@ti.func
def computeGradient(idx, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    if sumSigma < 1.0e-6:
        isSuccess = False

    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1
                                       ])  # (dcdS11, dcdS22)
    dsdx2 = ti.Vector([
        B[idx][0, 0] * U[0, 0] * V[0, 0] + B[idx][0, 1] * U[0, 0] * V[1, 0],
        B[idx][0, 0] * U[0, 1] * V[0, 1] + B[idx][0, 1] * U[0, 1] * V[1, 1]
    ])  #(ds11dx2, ds22dx2)
    dsdx3 = ti.Vector([
        B[idx][0, 0] * U[1, 0] * V[0, 0] + B[idx][0, 1] * U[1, 0] * V[1, 0],
        B[idx][0, 0] * U[1, 1] * V[0, 1] + B[idx][0, 1] * U[1, 1] * V[1, 1]
    ])  #(ds11dx3, ds22dx3)
    dsdx4 = ti.Vector([
        B[idx][1, 0] * U[0, 0] * V[0, 0] + B[idx][1, 1] * U[0, 0] * V[1, 0],
        B[idx][1, 0] * U[0, 1] * V[0, 1] + B[idx][1, 1] * U[0, 1] * V[1, 1]
    ])  #(ds11dx4, ds22dx4)
    dsdx5 = ti.Vector([
        B[idx][1, 0] * U[1, 0] * V[0, 0] + B[idx][1, 1] * U[1, 0] * V[1, 0],
        B[idx][1, 0] * U[1, 1] * V[0, 1] + B[idx][1, 1] * U[1, 1] * V[1, 1]
    ])  #(ds11dx5, ds22dx5)
    dsdx0 = -(dsdx2 + dsdx4)
    dsdx1 = -(dsdx3 + dsdx5)
    # s gradient
    dcdx0 = dcdS.dot(dsdx0)
    dcdx1 = dcdS.dot(dsdx1)
    dcdx2 = dcdS.dot(dsdx2)
    dcdx3 = dcdS.dot(dsdx3)
    dcdx4 = dcdS.dot(dsdx4)
    dcdx5 = dcdS.dot(dsdx5)

    g0 = ti.Vector([dcdx0, dcdx1])  # s gradient with respect to x0
    g1 = ti.Vector([dcdx2, dcdx3])  # s gradient with respect to x1
    g2 = ti.Vector([dcdx4, dcdx5])  # constraint gradient with respect to x2

    return g0, g1, g2, isSuccess


@ti.kernel
def semiEuler():
    # semi-Euler update pos & vel
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] += h * gravity + attractor_strength[None] * (
                attractor_pos[None] - pos[i]).normalized(1e-5)
            oldPos[i] = pos[i]
            pos[i] += h * vel[i]


@ti.kernel
def computeGradientVector():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        sumInvMass = invM0 + invM1 + invM2
        if sumInvMass < 1.0e-6:
            print("wrong invMass function")
        D_s = ti.Matrix.cols([b - a, c - a])
        F[i] = D_s @ B[i]
        U, S, V = ti.svd(F[i])
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        g0, g1, g2, isSuccess = computeGradient(i, U, S, V)
        if isSuccess:
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
            ) + invM2 * g2.norm_sqr()
            dLambda[i] = -(constraint + alpha * lagrangian[i]) / (l + alpha)
            lagrangian[i] = lagrangian[i] + dLambda[i]
            gradient[3 * i + 0] = g0
            gradient[3 * i + 1] = g1
            gradient[3 * i + 2] = g2


@ti.kernel
def updatePos():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        if (invM0 != 0.0):
            pos[ia] += omega * invM0 * dLambda[i] * gradient[3 * i + 0]
        if (invM1 != 0.0):
            pos[ib] += omega * invM1 * dLambda[i] * gradient[3 * i + 1]
        if (invM2 != 0.0):
            pos[ic] += omega * invM2 * dLambda[i] * gradient[3 * i + 2]


@ti.kernel
def updteVelocity():
    # update velocity
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.func
def computeConstriant(idx, x0, x1, x2):
    D_s = ti.Matrix.cols([x1 - x0, x2 - x0])
    F = D_s @ B[idx]
    U, S, V = ti.svd(F)
    constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    return constraint


@ti.kernel
def compute_C_and_gradC_kernel():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        sumInvMass = invM0 + invM1 + invM2
        if sumInvMass < 1.0e-6:
            print("wrong invMass function")
        D_s = ti.Matrix.cols([b - a, c - a])
        F[i] = D_s @ B[i]
        U, S, V = ti.svd(F[i])
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        constraints[i] = constraint
        g0, g1, g2, isSuccess = computeGradient(i, U, S, V)

        if isSuccess:
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
            ) + invM2 * g2.norm_sqr()
            dLambda[i] = -(constraint + alpha * lagrangian[i]) / (l + alpha)
            lagrangian[i] = lagrangian[i] + dLambda[i]
            gradient[3 * i + 0] = g0
            gradient[3 * i + 1] = g1
            gradient[3 * i + 2] = g2

            gradC[i, 0], gradC[i, 1], gradC[i, 2] = g0, g1, g2


@ti.kernel
def fill_gradC_np_kernel(
    A: ti.types.ndarray(),
    gradC: ti.template(),
    face_indices: ti.template(),
):
    for j in range(face_indices.shape[0]):
        ind = face_indices[j]
        for p in range(3):
            for d in range(2):
                pid = ind[p]
                A[j, 2 * pid + d] = gradC[j, p][d]





def solve_amg_my(A, b, x0, R, P, max_iter=1, r_norm_list=[]):
    max_levels = 2

    tol = 1e-3
    residuals = r_norm_list

    levels = []

    Level = namedtuple("Level", ["A", "R", "P", "presmoother", "postsmoother"])
    levels.append(Level(A, R, P, None, None))
    A2 = R @ A @ P
    levels.append(Level(A2, None, None, None, None))

    x = np.zeros_like(b)

    # Scale tol by normb
    # Don't scale tol earlier. The accel routine should also scale tol
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance

    # Start cycling (no acceleration)
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual
        # print(f"r:{normr:.2g}")

    b = np.ravel(b)
    x = np.ravel(x)

    it = 0

    while True:  # it <= max_iter and normr >= tol:
        if len(levels) == 1:
            # hierarchy has only 1 level
            # x = ml.coarse_solver(A, b)
            x = scipy.sparse.linalg.spsolve(A, b)
        else:
            __solve(levels, 0, x, b)

        it += 1

        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
            # print(f"r:{normr:.2g}")

        if normr < tol * normb:
            return x

        if it == max_iter:
            return x


def __solve(levels, lvl, x, b):
    A = levels[lvl].A

    # levels[lvl].presmoother(A, x, b)
    # x, _ = solve_gauss_seidel_symmetric(A, b, x, max_iterations=1)
    solve_symGS(A, x, b, iterations=1)
    
    # np.savetxt("r1_after_presmooth.txt", b - A @ x)

    residual = b - A @ x

    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    # if lvl == len(levels) - 2:
    #     # coarse_x[:] = coarse_solver(levels[-1].A, coarse_b)
    #     coarse_x[:] = scipy.sparse.linalg.spsolve(levels[-1].A, coarse_b)
    # else:
    #     ...

    x += levels[lvl].P @ coarse_x  # coarse grid correction

    # np.savetxt("r1_after_prolongate.txt", b - A @ x)

    # levels[lvl].postsmoother(A, x, b)
    # x, _ = solve_gauss_seidel_symmetric(A, b, x, max_iterations=1)
    solve_symGS(A, x, b, iterations=1)

    # np.savetxt("r1_after_postsmooth.txt", b - A @ x)


def solve_symGS(A, x, b, iterations=1, r_norm_list=[]):
    if not sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")
    
    r_norm = np.linalg.norm(A @ x - b)
    r_norm_list.append(r_norm)

    for _iter in range(iterations):
        # forward sweep
        # print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)
            r_norm = np.linalg.norm(A @ x - b)
            r_norm_list.append(r_norm)

        # backward sweep
        # print("backward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(
                A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
            )
            r_norm = np.linalg.norm(A @ x - b)
            r_norm_list.append(r_norm)
    return x


def amg_core_gauss_seidel(Ap, Aj, Ax, x, b, row_start: int, row_stop: int, row_step: int):
    for i in range(row_start, row_stop, row_step):
        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag


# ---------------------------------------------------------------------------- #
#                        All solvers refactored into one                       #
# ---------------------------------------------------------------------------- #
def substep_all_solver(max_iter=1, solver="DirectSolver", R=None, P=None):
    """
    ist: 要输入的instance: coarse or fine mesh\\
    max_iter: 最大迭代次数\\
    solver: 选择的solver, 可选项为: "Jacobi", "Gauss-Seidel", "SOR", "DirectSolver", "AMG"\\
    P: 粗网格到细网格的投影矩阵, 用于AMG, 默认为None, 除了AMG外都不需要\\
    R: 细网格到粗网格的投影矩阵, 用于AMG, 默认为None, 除了AMG外都不需要\\
    """
    # ist is instance of fine or coarse

    semiEuler()
    resetLagrangian()
    
    for ite in range(max_iter):
        t = time.time()

        # ----------------------------- prepare matrices ----------------------------- #
        print(f"----iter {ite}----")

        print("Assembling matrix")
        # copy pos to pos_mid
        pos_mid.from_numpy(pos.to_numpy())

        M = NF
        N = NV

        compute_C_and_gradC_kernel()

        # fill G matrix (gradC)
        G = np.zeros((M, 2 * N))
        fill_gradC_np_kernel(G, gradC, f2v)
        G = scipy.sparse.csr_matrix(G)

        # fill M_inv and ALPHA
        inv_mass_np = invMass.to_numpy()
        inv_mass_np = np.repeat(inv_mass_np, 2, axis=0)
        M_inv = scipy.sparse.diags(inv_mass_np)

        alpha_tilde_np = np.array([alpha] * M)
        ALPHA = scipy.sparse.diags(alpha_tilde_np)

        # assemble A and b
        print("Assemble A")
        A = G @ M_inv @ G.transpose() + ALPHA
        A = scipy.sparse.csr_matrix(A)
        b = -constraints.to_numpy() - alpha_tilde_np * lagrangian.to_numpy()

        print("Assemble matrix done")

        if output_A_flag:
            print("Save matrix to file")
            scipy.io.mmwrite("A.mtx", A)
            np.savetxt("b.txt", b)
            exit()

        # -------------------------------- solve Ax=b -------------------------------- #
        print("solve Ax=b")
        print(f"Solving by {solver}")

        dense = False
        if dense == True:
            A = np.asarray(A.todense())

        x0 = np.zeros_like(b)
        A = scipy.sparse.csr_matrix(A)


        if solver == "DirectSolver":
            print("DirectSolver")
            x = scipy.sparse.linalg.spsolve(A, b)
            print(f"r: {np.linalg.norm(A @ x - b):.2g}", )
        if solver == "GaussSeidel":
            print("GS")
            r_norm_list_GS = []
            t = perf_counter()
            x0 = np.zeros_like(b)
            x = np.zeros_like(b)
            r_norm = np.linalg.norm(A @ x - b)
            r_norm_list_GS.append(r_norm)
            for _ in range(5):
                amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
                r_norm = np.linalg.norm(A @ x - b)
                r_norm_list_GS.append(r_norm)
                print(f"{_} r:{r_norm:.2g}")
            t_GS = perf_counter() - t
            t = perf_counter()
        elif solver == "AMG":
            r_norm_list_amgmy = []
            x = solve_amg_my(A, b, x0, R, P, 5, r_norm_list_amgmy)
            for _ in range(len(r_norm_list_amgmy)):
                print(f"{_} r:{r_norm_list_amgmy[_]:.2g}")

        print(f"time of solve Ax=b: {time.time() - t:.2g}")

        # ------------------------- transfer data back to PBD ------------------------ #
        print("transfer data back to PBD")
        dlambda = x

        # lam += dlambda
        lagrangian.from_numpy(lagrangian.to_numpy() + dlambda)

        # dpos = M_inv @ G^T @ dlambda
        dpos = M_inv @ G.transpose() @ dlambda
        # pos+=dpos
        pos.from_numpy(pos_mid.to_numpy() + dpos.reshape(-1, 2))

    updteVelocity()


@ti.kernel
def compute_all_centroid(pos: ti.template(), f2v: ti.template(), res: ti.template()):
    for t in range(f2v.shape[0]):
        a, b, c = f2v[t]
        p0, p1, p2 = pos[a], pos[b], pos[c]
        p = (p0 + p1 + p2) / 4
        res[t] = p



def compute_R_and_P_kmeans():
    print(">>Computing P and R...")
    t = time.time()

    from scipy.cluster.vq import vq, kmeans, whiten

    # 计算所有四面体的质心
    print(">>Computing all tet centroid...")
    compute_all_centroid(pos, f2v, centroids)

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = centroids.to_numpy()

    np.savetxt("centroid.txt", input)

    N = input.shape[0]
    k = int(N / 10)
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

    # TODO
    compute_R_based_on_kmeans_label(labels, R)

    R = scipy.sparse.csr_matrix(R)
    P = R.transpose()
    print(f"Computing P and R done, time = {time.time() - t}")

    # print(f"writing P and R...")
    # scipy.io.mmwrite("R.mtx", R)
    # scipy.io.mmwrite("P.mtx", P)
    # print(f"writing P and R done")

    return R, P


@ti.kernel
def compute_R_based_on_kmeans_label(
    labels: ti.types.ndarray(dtype=int),
    R: ti.types.ndarray(),
):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if labels[j] == i:
                R[i, j] = 1


# def plot_r_norm_list(data, ax, title):
#     x = np.arange(len(data))
#     ax.plot(x, data, "-o")
#     ax.set_title(title)
#     # ax.set_yscale("log")
#     ax.set_xlabel("iteration")
#     ax.set_ylabel("residual")



init_mesh()
init_pos()

R, P = compute_R_and_P_kmeans()

pause = False
gui = ti.GUI('XPBD-FEM')
first = True
drawTime = 0
sumTime = 0
frame_num = 0
end_frame = 100

while gui.running:
    frame_num += 1
    print("\n\nframe_num: ", frame_num)
    realStart = time.time()
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            pause = not pause
        elif e.key == 'o':
            output_A_flag = True
            print("output_A_flag: ", output_A_flag)


    
    # mouse_pos = gui.get_cursor_pos()
    mouse_pos = [0,0]
    attractor_pos[None] = mouse_pos
    # attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed( gui.RMB)
    attractor_strength[None] = 1

    gui.circle(mouse_pos, radius=15, color=0x336699)



    if not pause:
        for i in range(NumSteps):
            # substep_all_solver(1,"DirectSolver")
            # substep_all_solver(1,"GaussSeidel")
            substep_all_solver(1,"AMG", R, P)
            # semiEuler()
            # resetLagrangian()
            # for ite in range(MaxIte):
            #     computeGradientVector()
            #     updatePos()
            # updteVelocity()
    ti.sync()
    start = time.time()
    faces = f2v.to_numpy()
    for i in range(NF):
        ia, ib, ic = faces[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        gui.triangle(a, b, c, color=0x00FF00)

    positions = pos.to_numpy()
    gui.circles(positions, radius=2, color=0x0000FF)
    for i in range(N + 1):
        k = i * (N + 1) + N
        staticVerts = positions[k]
        gui.circle(staticVerts, radius=5, color=0xFF0000)

    filename = f'result/iter1/{frame_num:04d}.png'   # create filename with suffix png
    print(f'Frame {frame_num} is recorded in {filename}')
    gui.show(filename)  # export and show in GUI

    end = time.time()
    drawTime = (end - start)
    sumTime = (end - realStart)
    print("Draw Time Ratio; ", drawTime / sumTime)

    if frame_num == end_frame:
        break

# ti.print_kernel_profile_info()