import taichi as ti
import numpy as np
import time
import scipy
from pathlib import Path
import os
from matplotlib import pyplot as plt

ti.init(arch=ti.cpu, debug=True)
N = 256
NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f32, shape=NV)
acc_pos = ti.Vector.field(3, ti.f32, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edge = ti.Vector.field(2, ti.i32, shape=NE)

old_pos = ti.Vector.field(3, ti.f32, NV)
inv_mass = ti.field(ti.f32, NV)
vel = ti.Vector.field(3, ti.f32, NV)
rest_len = ti.field(ti.f32, NE)
h = 0.01

paused = ti.field(ti.i32, shape=())

NF=NT
compliance = 1.0e-8  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h)  # timestep related compliance, see XPBD paper
lagrangian = ti.field(float, NE)  # lagrangian multipliers
pos_mid = ti.Vector.field(3, float, NV)  # mid pos
constraints = ti.field(float, NE)
gradC = ti.Vector.field(3, float, (NE,2))  # gradient of constraints
e2v = ti.Vector.field(3, int, NE)  # ids of three vertices of each edge
dLambda = ti.field(float, NE)


new_M = int(NE / 100)
y_jprime = np.zeros((new_M),float)
numerator = np.zeros((NE),float)
denominator = np.zeros((NE),float)
numerator_lumped = np.zeros((new_M),float)
denominator_lumped = np.zeros((new_M),float)
edge_center = np.zeros((NE,3),float)

minus_C_minus_alpha_lambda = ti.field(float, NE)


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, 0.5, j / N])
        inv_mass[idx] = 1.0
    inv_mass[N] = 0.0
    inv_mass[NV-1] = 0.0


@ti.kernel
def init_tri():
    for i, j in ti.ndrange(N, N):
        tri_idx = 6 * (i * N + j)
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 2
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2
        else:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 1
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx + 1
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2


@ti.kernel
def init_edge():
    for i, j in ti.ndrange(N + 1, N):
        edge_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edge_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()

@ti.kernel
def init_edge_center(edge_center:ti.types.ndarray(dtype=ti.math.vec3)):
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        edge_center[i] = (p1 + p2) / 2.0


@ti.kernel
def semi_euler():
    gravity = ti.Vector([0.0, -0.1, 0.0])
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            old_pos[i] = pos[i]
            pos[i] += h * vel[i]


@ti.kernel
def solve_constraints():
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        if invM0 != 0.0:
            acc_pos[idx0] += invM0 * l * gradient
        if invM1 != 0.0:
            acc_pos[idx1] -= invM1 * l * gradient


@ti.kernel
def solve_subspace_constraints_xpbd(
    labels: ti.types.ndarray(dtype=int),
    numerator: ti.types.ndarray(dtype=float),
    denominator: ti.types.ndarray(dtype=float),
    numerator_lumped: ti.types.ndarray(dtype=float),
    denominator_lumped: ti.types.ndarray(dtype=float),
    y_jprime: ti.types.ndarray(dtype=float),
):
    #subspace solving
    # ti.loop_config(serialize=True)
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        numerator[i] = -(constraint + lagrangian[i] * alpha)
        denominator[i] = invM0 + invM1 + alpha

    for i in range(new_M):
        numerator_lumped[i] = 0.0
        denominator_lumped[i] = 0.0
    for i in range(NE):
        jp = labels[i]
        numerator_lumped[jp] += numerator[i]
        denominator_lumped[jp] += denominator[i]
    for jp in range(new_M):
        y_jprime[jp] = numerator_lumped[jp] / denominator_lumped[jp]
    
    # prolongation
    for i in range(NE):
        jp = labels[i]
        dLambda[i] =  y_jprime[jp]
        lagrangian[i] += dLambda[i]
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        gradient = dis.normalized()
        if invM0 != 0.0:
            acc_pos[idx0] += invM0 * dLambda[i] * gradient
        if invM1 != 0.0:
            acc_pos[idx1] -= invM1 * dLambda[i] * gradient

@ti.kernel
def solve_constraints_xpbd():
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian[i] * alpha) / (invM0 + invM1 + alpha)
        lagrangian[i] += delta_lagrangian

        # residual
        minus_C_minus_alpha_lambda[i] = -(constraint + alpha * lagrangian[i])
        
        if invM0 != 0.0:
            acc_pos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            acc_pos[idx1] -= invM1 * delta_lagrangian * gradient

@ti.kernel
def update_pos():
    for i in range(NV):
        if inv_mass[i] != 0.0:
            pos[i] += 0.5 * acc_pos[i]

@ti.kernel
def update_vel():
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h

@ti.kernel 
def collision():
    for i in range(NV):
        if pos[i][2] < -2.0:
            pos[i][2] = 0.0

@ti.kernel 
def reset_accpos():
    for i in range(NV):
        acc_pos[i] = ti.Vector([0.0, 0.0, 0.0])

def step_pbd(max_iter):
    semi_euler()
    for i in range(max_iter):
        reset_accpos()
        solve_constraints()
        update_pos()
        collision()
    update_vel()


@ti.kernel
def calc_residual():
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]

        # residual(lagrangian=0 for first iteration)
        minus_C_minus_alpha_lambda[i] = -(constraint + alpha * lagrangian[i])


def step_xpbd(max_iter):
    semi_euler()
    reset_lagrangian()

    residual = np.zeros((max_iter+1),float)
    calc_residual()
    residual[0] = np.linalg.norm(minus_C_minus_alpha_lambda.to_numpy())

    for i in range(max_iter):
        reset_accpos()
        # solve_subspace_constraints_xpbd(labels, numerator, denominator, numerator_lumped, denominator_lumped, y_jprime)
        solve_constraints_xpbd()
        update_pos()
        collision()

        residual[i+1] = np.linalg.norm(minus_C_minus_alpha_lambda.to_numpy())
    np.savetxt(out_dir + f"residual_{frame_num}.txt",residual)

    update_vel()



# ---------------------------------------------------------------------------- #
#                                build hierarchy                               #
# ---------------------------------------------------------------------------- #

@ti.kernel
def compute_all_centroid(pos: ti.template(), tet_indices: ti.template(), res: ti.template()):
    for t in range(tet_indices.shape[0]):
        a, b, c, d = tet_indices[t]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        p = (p0 + p1 + p2 + p3) / 4
        res[t] = p

@ti.kernel
def compute_R_based_on_kmeans_label_triplets(
    labels: ti.types.ndarray(dtype=int),
    ii: ti.types.ndarray(dtype=int),
    jj: ti.types.ndarray(dtype=int),
    vv: ti.types.ndarray(dtype=int),
    new_M: ti.i32,
    M: ti.i32
):
    cnt=0
    ti.loop_config(serialize=True)
    for i in range(new_M):
        for j in range(M):
            if labels[j] == i:
                ii[cnt],jj[cnt],vv[cnt] = i,j,1
                cnt+=1



def compute_R_and_P_kmeans():
    print(">>Computing P and R...")
    t = time.time()

    from scipy.cluster.vq import vq, kmeans, whiten

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = edge_center

    M = int(input.shape[0])
    new_M = int(input.shape[0] / 100)
    print("M: ", M)
    print("new_M: ", new_M)

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
    i_arr = np.zeros((M), dtype=np.int32)
    j_arr = np.zeros((M), dtype=np.int32)
    v_arr = np.zeros((M), dtype=np.int32)
    compute_R_based_on_kmeans_label_triplets(labels, i_arr, j_arr, v_arr, new_M, M)

    R = scipy.sparse.coo_array((v_arr, (i_arr, j_arr)), shape=(new_M, M)).tocsr()
    P = R.transpose()
    print(f"Computing P and R done, time = {time.time() - t}")

    # print(f"writing P and R...")
    # scipy.io.mmwrite("R.mtx", R)
    # scipy.io.mmwrite("P.mtx", P)
    # print(f"writing P and R done")

    return R, P, labels, new_M

# ---------------------------------------------------------------------------- #
#                                   for ours                                   #
# ---------------------------------------------------------------------------- #

@ti.kernel
def compute_C_and_gradC_kernel():
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient_dist = dis.normalized()

        gradC[i, 0], gradC[i, 1] = gradient_dist, -gradient_dist
        constraints[i] = constraint


@ti.kernel
def fill_gradC_np_kernel(
    A: ti.types.ndarray(),
    gradC: ti.template(),
    e2v: ti.template(),
):
    for j in e2v:
        ind = e2v[j]
        for p in range(2): #which point in the edge
            for d in range(3): #which dimension
                pid = ind[p]
                A[j, 3 * pid + d] = gradC[j, p][d]



@ti.kernel
def fill_gradC_csr_kernel(
    A_indptr: ti.types.ndarray(dtype=int),
    A_indices: ti.types.ndarray(dtype=int),
    A_data: ti.types.ndarray(dtype=float),
    gradC: ti.template(),
    e2v: ti.template(),
):
    '''
    fill CSR format sparse matrix A with gradC
    CSR format: The column indices for row i are stored in indices[indptr[i]:indptr[i+1]] 
    and their corresponding values are stored in data[indptr[i]:indptr[i+1]]
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
    '''
    A_indptr[0] = 0
    for i in e2v:
        ind = e2v[i]
        A_indptr[i + 1] = i * 6 + 6
        A_indices[i * 6 + 0] = 3 * ind[0] + 0
        A_indices[i * 6 + 1] = 3 * ind[0] + 1
        A_indices[i * 6 + 2] = 3 * ind[0] + 2
        A_indices[i * 6 + 3] = 3 * ind[1] + 0
        A_indices[i * 6 + 4] = 3 * ind[1] + 1
        A_indices[i * 6 + 5] = 3 * ind[1] + 2
        A_data[i * 6 + 0] = gradC[i, 0][0]
        A_data[i * 6 + 1] = gradC[i, 0][1]
        A_data[i * 6 + 2] = gradC[i, 0][2]
        A_data[i * 6 + 3] = gradC[i, 1][0]
        A_data[i * 6 + 4] = gradC[i, 1][1]
        A_data[i * 6 + 5] = gradC[i, 1][2]
        # for p in range(2): #which point in the edge
        #     for d in range(3): #which dimension
        #         pid = ind[p]
        #         A_indices[i * 6 + p * 3 + d] = 3 * pid + d
        #         A_data[i * 6 + p * 3 + d] = gradC[i, p][d]
        # A_indptr[i + 1] = i * 6 + 6


@ti.kernel
def fill_gradC_triplets_kernel(
    ii:ti.types.ndarray(dtype=ti.i32),
    jj:ti.types.ndarray(dtype=ti.i32),
    vv:ti.types.ndarray(dtype=ti.f32),
    gradC: ti.template(),
    tet_indices: ti.template(),
):
    cnt=0
    ti.loop_config(serialize=True)
    for j in range(tet_indices.shape[0]):
        ind = tet_indices[j]
        for p in range(4):
            for d in range(3):
                pid = ind[p]
                ii[cnt],jj[cnt],vv[cnt] = j, 3 * pid + d, gradC[j, p][d]
                cnt+=1

@ti.kernel
def reset_lagrangian():
    for i in range(NE):
        lagrangian[i] = 0.0


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


def amg_core_gauss_seidel_kernel(Ap: ti.types.ndarray(dtype=int),
                                 Aj: ti.types.ndarray(dtype=int),
                                 Ax: ti.types.ndarray(dtype=float),
                                 x: ti.types.ndarray(),
                                 b: ti.types.ndarray(),
                                 row_start: int,
                                 row_stop: int,
                                 row_step: int):
    if row_step < 0:
        assert "row_step must be positive"
    for i in range(row_start, row_stop):
        if i%row_step != 0:
            continue

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


def substep_all_solver(max_iter=1, solver="DirectSolver", R=None, P=None):
    """
    max_iter: 最大迭代次数\\
    solver: 选择的solver, 可选项为: "Jacobi", "Gauss-Seidel", "SOR", "DirectSolver", "AMG"\\
    P: 粗网格到细网格的投影矩阵, 用于AMG, 默认为None, 除了AMG外都不需要\\
    R: 细网格到粗网格的投影矩阵, 用于AMG, 默认为None, 除了AMG外都不需要\\
    """
    # ist is instance of fine or coarse

    semi_euler()
    reset_lagrangian()
    
    for ite in range(max_iter):
        t = time.time()

        # ----------------------------- prepare matrices ----------------------------- #
        print(f"----iter {ite}----")

        print("Assembling matrix")
        # copy pos to pos_mid
        pos_mid.from_numpy(pos.to_numpy())

        M = NE
        N = NV

        compute_C_and_gradC_kernel()

        # fill G matrix (gradC)
        G = np.zeros((M, 3 * N))
        fill_gradC_np_kernel(G, gradC, e2v)
        G = scipy.sparse.csr_matrix(G)
        # print("writing G.mtx")
        # scipy.io.mmwrite("G.mtx", G)

        # G1_indptr = np.zeros((M+1),int)
        # G1_indices = np.zeros((6*M),int)
        # G1_data = np.zeros((6*M),float)
        # fill_gradC_csr_kernel(G1_indptr, G1_indices, G1_data, gradC, e2v)
        # G1 = scipy.sparse.csr_matrix((G1_data, G1_indices, G1_indptr),shape=(M, 3*N))
        # print("writing G1.mtx")
        # scipy.io.mmwrite("G1.mtx", G1)
        # exit()

        # fill M_inv and ALPHA
        inv_mass_np = inv_mass.to_numpy()
        inv_mass_np = np.repeat(inv_mass_np, 3, axis=0)
        M_inv = scipy.sparse.diags(inv_mass_np)

        alpha_tilde_np = np.array([alpha] * M)
        ALPHA = scipy.sparse.diags(alpha_tilde_np)

        # assemble A and b
        print("Assemble A")
        A = G @ M_inv @ G.transpose() + ALPHA
        A = scipy.sparse.csr_matrix(A)
        b = -constraints.to_numpy() - alpha_tilde_np * lagrangian.to_numpy()

        print("Assemble matrix done")
        print("A:", A.shape, " b:", b.shape)

        # scipy.io.mmwrite("A.mtx", A)
        # plt.spy(A, markersize=1)
        # plt.show()
        # exit()

        # -------------------------------- solve Ax=b -------------------------------- #
        print("solve Ax=b")
        print(f"Solving by {solver}")

        t = time.perf_counter()

        x0 = np.zeros_like(b)

        r_norm_list = []
        r_norm_list.append(np.linalg.norm(A @ x0 - b)) # first residual
        
        if solver == "DirectSolver":
            print("DirectSolver")
            x = scipy.sparse.linalg.spsolve(A, b)
            print(f"r: {np.linalg.norm(A @ x - b):.2g}" )

        if solver == "GaussSeidel":
            print("GaussSeidel")
            x = np.zeros_like(b)
            for _ in range(1):
                # amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
                amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
                r_norm = np.linalg.norm(A @ x - b)
                r_norm_list.append(r_norm)
                print(f"{_} r:{r_norm:.2g}")
            np.savetxt(out_dir + f"residual_frame_{frame_num}.txt",np.array(r_norm_list))

        elif solver == "AMG":
            x = solve_amg_my(A, b, x0, R, P, 1, r_norm_list)
            for _ in range(len(r_norm_list)):
                print(f"{_} r:{r_norm_list[_]:.2g}")
            np.savetxt(out_dir + f"residual_frame_{frame_num}.txt",np.array(r_norm_list))

        print(f"time of Ax=b: {(time.perf_counter() - t):.2g}")

        # ------------------------- transfer data back to pos ------------------------ #
        print("transfer data back to pos")
        dlambda = x

        # lam += dlambda
        lagrangian.from_numpy(lagrangian.to_numpy() + dlambda)

        # dpos = M_inv @ G^T @ dlambda
        dpos = M_inv @ G.transpose() @ dlambda
        # pos+=dpos
        pos.from_numpy(pos_mid.to_numpy() + dpos.reshape(-1, 3))

    update_vel()

    return r_norm_list


def mkdir_if_not_exist(path=None):
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)



frame_num = 0
end_frame = 1000
# out_dir = f"./result/cloth3d_fullspace/"
out_dir = f"./result/cloth3d_subspace/"
mkdir_if_not_exist(out_dir)
save_image = True
max_iter = 50

init_pos()
init_tri()
init_edge()
init_edge_center(edge_center)
# np.savetxt("edge_center.txt", edge_center)
R, P, labels, new_M = compute_R_and_P_kmeans()
# np.savetxt("labels.txt", labels)
# np.loadtxt("labels.txt", labels)

window = ti.ui.Window("Display Mesh", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 0.0, 2.5)
camera.lookat(0.5, 0.5, 0.0)
camera.fov(90)
gui = window.get_gui()

paused[None] = 0
while window.running:
    frame_num += 1
    print("\n\nframe_num: ", frame_num)

    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            exit()
        if e.key == ti.ui.SPACE:
            paused[None] = not paused[None]
            print("paused:",paused[None])

    if not paused[None]:
        # step_pbd(max_iter)
        step_xpbd(max_iter)
        # substep_all_solver(max_iter=max_iter, solver="GaussSeidel")
    
    camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

    scene.mesh(pos, tri, color=(1.0,0,0), two_sided=True)
    # scene.particles(pos, radius=0.01, color=(0.6,0.0,0.0))
    canvas.scene(scene)

    # you must call this function, even if we just want to save the image, otherwise the GUI image will not update.
    window.show()

    # if save_image and frame_num % 10 == 0:
    file_path = out_dir + f"{frame_num:04d}.png"
    window.save_image(file_path)  # export and show in GUI

    if frame_num == end_frame:
        break
