import taichi as ti
import numpy as np
import time
import scipy
import scipy.sparse as sp
from pathlib import Path
import os
from matplotlib import pyplot as plt
import shutil, glob
import meshio
import tqdm
import argparse

ti.init(arch=ti.cpu)

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=64)

N = parser.parse_args().N
print("N: ", N)

NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
h = 0.01
M = NE
new_M = int(NE / 100)
compliance = 1.0e-8  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h)  # timestep related compliance, see XPBD paper
omega = 0.5

tri = ti.field(ti.i32, shape=3 * NT)
pos_ti = ti.Vector.field(3, ti.f32, shape=NV)


edge        = ti.Vector.field(2, dtype=int, shape=(NE))
pos         = ti.Vector.field(3, dtype=float, shape=(NV))
acc_pos     = ti.Vector.field(3, dtype=float, shape=(NV))
old_pos     = ti.Vector.field(3, dtype=float, shape=(NV))
vel         = ti.Vector.field(3, dtype=float, shape=(NV))
pos_mid     = ti.Vector.field(3, dtype=float, shape=(NV))
inv_mass    = ti.field(dtype=float, shape=(NV))
rest_len    = ti.field(dtype=float, shape=(NE))
lagrangian  = ti.field(dtype=float, shape=(NE))  
constraints = ti.field(dtype=float, shape=(NE))  
dLambda     = ti.field(dtype=float, shape=(NE))
numerator   = ti.field(dtype=float, shape=(NE))
denominator = ti.field(dtype=float, shape=(NE))
gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(NE,2)) 
edge_center = ti.Vector.field(3, dtype = ti.float32, shape=(NE))
# y_jprime    = ti.field(shape=(new_M),   dtype = ti.float32)
# numerator_lumped    = ti.field(shape=(new_M), dtype = ti.float32)
# denominator_lumped  = ti.field(shape=(new_M), dtype = ti.float32)
dual_residual       = ti.field(shape=(NE),    dtype = ti.float32) # -C - alpha * lagrangian
adjacent_edge_abc   = ti.field(shape=(NE,42),  dtype = ti.int32)
adjacent_edge   = ti.field(shape=(NE,14),  dtype = ti.int32)

@ti.kernel
def init_pos(
    inv_mass:ti.template(),
    pos:ti.template(),
):
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        # pos[idx] = ti.Vector([i / N,  j / N, 0.5])  # vertical hang
        pos[idx] = ti.Vector([i / N, 0.5, j / N]) # horizontal hang
        inv_mass[idx] = 1.0
    inv_mass[N] = 0.0
    inv_mass[NV-1] = 0.0


@ti.kernel
def init_tri(tri:ti.template()):
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
def init_edge(
    edge:ti.template(),
    rest_len:ti.template(),
    pos:ti.template(),
):
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
def init_edge_center(
    edge_center:ti.template(),
    edge:ti.template(),
    pos:ti.template(),
):
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        edge_center[i] = (p1 + p2) / 2.0

def init_adjacent_edge_abc():
    num_adjacent_edge_np = np.loadtxt(misc_dir_path+"num_adjacent_edge.txt", dtype=np.int32)
    MAX = max(num_adjacent_edge_np) #14
    
    filename = misc_dir_path+"adjacent_edge_abc.txt"
    def pad_list(lst, padding, default=-1):
        return lst + (padding - len(lst))*[default]
    with open(filename,"r") as f:
        all_data=(map(int, x.split()) for x in f)
        a = np.array([pad_list(list(x), MAX*3) for x in all_data])
    adjacent_edge_abc.from_numpy(a)

    filename = misc_dir_path+"adjacent_edge.txt"
    with open(filename,"r") as f:
        all_data=(map(int, x.split()) for x in f)
        b = np.array([pad_list(list(x), MAX) for x in all_data])
    adjacent_edge.from_numpy(b)
    ...

@ti.kernel
def semi_euler(
    old_pos:ti.template(),
    inv_mass:ti.template(),
    vel:ti.template(),
    pos:ti.template(),
):
    gravity = ti.Vector([0.0, -0.1, 0.0])
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            old_pos[i] = pos[i]
            pos[i] += h * vel[i]


@ti.kernel
def solve_constraints(
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    acc_pos:ti.template(),
    pos:ti.template(),
):
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
    labels: ti.template(),
    numerator: ti.template(),
    denominator: ti.template(),
    numerator_lumped: ti.template(),
    denominator_lumped: ti.template(),
    y_jprime: ti.template(),
    dLambda: ti.template(),
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    acc_pos:ti.template(),
    pos:ti.template(),
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
def solve_constraints_xpbd(
    dual_residual: ti.template(),
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    acc_pos:ti.template(),
    pos:ti.template(),
):
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
        dual_residual[i] = -(constraint + alpha * lagrangian[i])
        
        if invM0 != 0.0:
            acc_pos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            acc_pos[idx1] -= invM1 * delta_lagrangian * gradient

@ti.kernel
def update_pos(
    inv_mass:ti.template(),
    acc_pos:ti.template(),
    pos:ti.template(),
):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            pos[i] += omega * acc_pos[i]

@ti.kernel
def update_vel(
    old_pos:ti.template(),
    inv_mass:ti.template(),    
    vel:ti.template(),
    pos:ti.template(),
):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h

@ti.kernel 
def collision(pos:ti.template()):
    for i in range(NV):
        if pos[i][2] < -2.0:
            pos[i][2] = 0.0

@ti.kernel 
def reset_accpos(acc_pos:ti.template()):
    for i in range(NV):
        acc_pos[i] = ti.Vector([0.0, 0.0, 0.0])



@ti.kernel
def calc_dual_residual(
    dual_residual: ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    pos:ti.template(),
):
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]

        # residual(lagrangian=0 for first iteration)
        dual_residual[i] = -(constraint + alpha * lagrangian[i])


def step_xpbd(max_iter):
    semi_euler(old_pos, inv_mass, vel, pos)
    reset_lagrangian(lagrangian)

    residual = np.zeros((max_iter+1),float)
    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    residual[0] = np.linalg.norm(dual_residual.to_numpy())

    for i in range(max_iter):
        reset_accpos(acc_pos)
        # solve_subspace_constraints_xpbd(labels, numerator, denominator, numerator_lumped, denominator_lumped, y_jprime, dLambda, inv_mass, edge, rest_len, lagrangian, acc_pos, pos)
        solve_constraints_xpbd(dual_residual, inv_mass, edge, rest_len, lagrangian, acc_pos, pos)
        update_pos(inv_mass, acc_pos, pos)
        collision(pos)

        residual[i+1] = np.linalg.norm(dual_residual.to_numpy())
    np.savetxt(out_dir + f"dual_residual_{frame_num}.txt",residual)

    update_vel(old_pos, inv_mass, vel, pos)



# ---------------------------------------------------------------------------- #
#                                build hierarchy                               #
# ---------------------------------------------------------------------------- #
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
    t = time.perf_counter()

    from scipy.cluster.vq import vq, kmeans, whiten

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = edge_center.to_numpy()

    M = NE
    global new_M
    print("M: ", M, "  new_M: ", new_M)

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
    print(f"Computing P and R done, time = {time.perf_counter() - t}")

    # print(f"writing P and R...")
    # scipy.io.mmwrite("R.mtx", R)
    # scipy.io.mmwrite("P.mtx", P)
    # print(f"writing P and R done")

    return R, P, labels, new_M

# ---------------------------------------------------------------------------- #
#                                   for ours                                   #
# ---------------------------------------------------------------------------- #
@ti.kernel
def compute_C_and_gradC_kernel(
    pos:ti.template(),
    gradC: ti.template(),
    edge:ti.template(),
    constraints:ti.template(),
    rest_len:ti.template(),
):
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]
        g = dis.normalized()

        gradC[i, 0] = g
        gradC[i, 1] = -g



@ti.kernel
def fill_gradC_triplets_kernel(
    ii:ti.types.ndarray(dtype=ti.i32),
    jj:ti.types.ndarray(dtype=ti.i32),
    vv:ti.types.ndarray(dtype=ti.i32),
    gradC: ti.template(),
    edge: ti.template(),
):
    cnt=0
    ti.loop_config(serialize=True)
    for j in range(edge.shape[0]):
        ind = edge[j]
        for p in range(2):
            for d in range(3):
                pid = ind[p]
                ii[cnt],jj[cnt],vv[cnt] = j, 3 * pid + d, gradC[j, p][d]
                cnt+=1



@ti.kernel
def fill_gradC_np_kernel(
    G: ti.types.ndarray(),
    gradC: ti.template(),
    edge: ti.template(),
):
    for j in edge:
        ind = edge[j]
        for p in range(2): #which point in the edge
            for d in range(3): #which dimension
                pid = ind[p]
                G[j, 3 * pid + d] = gradC[j, p][d]


@ti.kernel
def reset_lagrangian(lagrangian: ti.template()):
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

@ti.kernel
def amg_core_gauss_seidel_kernel(Ap: ti.types.ndarray(),
                                 Aj: ti.types.ndarray(),
                                 Ax: ti.types.ndarray(),
                                 x: ti.types.ndarray(),
                                 b: ti.types.ndarray(),
                                 row_start: int,
                                 row_stop: int,
                                 row_step: int):
    # if row_step < 0:
    #     assert "row_step must be positive"
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


def solve_amg(A, b, x0, R, P):
    tol = 1e-3
    residuals = []
    maxiter = 1
    A2 = R @ A @ P
    x = x0
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0  # set so that we have an absolute tolerance
    normr = np.linalg.norm(b - A @ x)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual
    b = np.ravel(b)
    x = np.ravel(x)
    it = 0
    while True:  # it <= maxiter and normr >= tol:
        residual = b - A @ x
        coarse_b = R @ residual  # restriction
        coarse_x = np.zeros_like(coarse_b)
        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)
        x += P @ coarse_x 
        # amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
        it += 1
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)
        if normr < tol * normb:
            return x
        if it == maxiter:
            return x

def transfer_back_to_pos_matrix(x, M_inv, G, pos_mid):
    dLambda_ = x.copy()
    lagrangian.from_numpy(lagrangian.to_numpy() + dLambda_)
    dpos = M_inv @ G.transpose() @ dLambda_
    dpos = dpos.reshape(-1, 3)
    pos.from_numpy(pos_mid.to_numpy() + dpos)

@ti.kernel
def transfer_back_to_pos_mfree_kernel():
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        delta_lagrangian = dLambda[i]
        lagrangian[i] += delta_lagrangian

        gradient = gradC[i, 0]
        
        if invM0 != 0.0:
            acc_pos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            acc_pos[idx1] -= invM1 * delta_lagrangian * gradient


def transfer_back_to_pos_mfree(x):
    dLambda.from_numpy(x)
    reset_accpos(acc_pos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(inv_mass, acc_pos, pos)
    collision(pos)

def spy_A(A,b):
    print("A:", A.shape, " b:", b.shape)
    scipy.io.mmwrite("A.mtx", A)
    plt.spy(A, markersize=1)
    plt.show()
    exit()

def substep_all_solver(max_iter=1, solver_type="Direct", R=None, P=None):
    global pos, lagrangian
    semi_euler(old_pos, inv_mass, vel, pos)
    reset_lagrangian(lagrangian)
    inv_mass_np = np.repeat(inv_mass.to_numpy(), 3, axis=0)
    M_inv = scipy.sparse.diags(inv_mass_np)
    alpha_tilde_np = np.array([alpha] * M)
    ALPHA = scipy.sparse.diags(alpha_tilde_np)
    for ite in range(max_iter):
        copy_field(pos_mid, pos)
        G_ii, G_jj, G_vv = np.zeros(M*6, dtype=np.int32), np.zeros(M*6, dtype=np.int32), np.zeros(M*6, dtype=np.float32)
        compute_C_and_gradC_kernel(pos, gradC, edge, constraints, rest_len)
        fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, edge)
        G = scipy.sparse.csr_array((G_vv, (G_ii, G_jj)), shape=(M, 3 * NV))
        A = G @ M_inv @ G.transpose() + ALPHA
        A = scipy.sparse.csr_matrix(A)
        b = -constraints.to_numpy() - alpha_tilde_np * lagrangian.to_numpy()

        if frame_num == stop_frame and export_matrix:
            print(f"writting A and b to {out_dir}")
            scipy.io.mmwrite(out_dir + f"A.mtx", A)
            np.savetxt(out_dir + f"b.txt", b)
            exit()
        
        x0 = np.zeros_like(b)
        if solver_type == "Direct":
            x = scipy.sparse.linalg.spsolve(A, b)
        elif solver_type == "GS":
            x = np.zeros_like(b)
            for _ in range(1):
                amg_core_gauss_seidel_kernel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x0)), row_step=1)
        elif solver_type == "AMG":
            x = solve_amg(A, b, x0, R, P)
        
        transfer_back_to_pos_mfree(x)

        if export_residual:
            r_norm = np.linalg.norm(A @ x - b)
            calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
            dual_r = np.linalg.norm(dual_residual.to_numpy()).astype(np.float32)
            with open(out_dir+f"r_frame_{frame_num}.txt", 'a+') as f:
                f.write(f"{r_norm}\n")
            with open(out_dir+f"dual_r_frame_{frame_num}.txt", 'a+') as f:
                f.write(f"{dual_r}\n")
    update_vel(old_pos, inv_mass, vel, pos)


def mkdir_if_not_exist(path=None):
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)

def delete_txt_files(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*r_frame_*.txt'))
    for file_path in txt_files:
        os.remove(file_path)

def clean_result_dir(folder_path):
    print(f"clean {folder_path}...")
    to_remove = []
    for name in [
        '*.txt',
        '*.obj',
        '*.png',
        '*.ply'
    ]:
        files = glob.glob(os.path.join(folder_path, name))
        to_remove += (files)
    print(f"removing {len(to_remove)} files")
    for file_path in to_remove:
        os.remove(file_path)
    print(f"clean {folder_path} done")

@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]


def write_obj(filename, pos, tri):
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )
    mesh.write(filename)
    return mesh

frame_num = 0
end_frame = 1000
out_dir = f"./result/test/"
proj_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("proj_dir_path: ", proj_dir_path)
misc_dir_path = proj_dir_path + "/data/misc/"
mkdir_if_not_exist(out_dir)
clean_result_dir(out_dir)
save_image = True
max_iter = 50
paused = False
save_P, load_P = False, True
use_viewer = False
export_obj = True
export_residual = False
solver_type = "GS" # "AMG", "GS", "XPBD"
export_matrix = True
stop_frame = 10

timer_all = time.perf_counter()
init_pos(inv_mass,pos)
init_tri(tri)
init_edge(edge, rest_len, pos)
if solver_type=="AMG":
    init_edge_center(edge_center, edge, pos)
    init_adjacent_edge_abc()

    if save_P:
        R, P, labels, new_M = compute_R_and_P_kmeans()
        scipy.io.mmwrite(misc_dir_path + "R.mtx", R)
        scipy.io.mmwrite(misc_dir_path + "P.mtx", P)
        np.savetxt(misc_dir_path + "labels.txt", labels, fmt="%d")
    if load_P:
        R = scipy.io.mmread(misc_dir_path+ "R.mtx")
        P = scipy.io.mmread(misc_dir_path+ "P.mtx")
        # labels = np.loadtxt( "labels.txt", dtype=np.int32)

class Viewer:
    if use_viewer:
        window = ti.ui.Window("Display Mesh", (1024, 1024))
        canvas = window.get_canvas()
        canvas.set_background_color((1, 1, 1))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        # camera.position(0.5, 0.4702609, 1.52483202)
        # camera.lookat(0.5, 0.9702609, -0.97516798)
        camera.position(0.5, 0.0, 2.5)
        camera.lookat(0.5, 0.5, 0.0)
        camera.fov(90)
        gui = window.get_gui()

viewer = Viewer()

step_pbar = tqdm.tqdm(total=end_frame)
while True:
    step_pbar.update(1)
    time_one_frame = time.perf_counter()
    frame_num += 1
    if use_viewer:
        for e in viewer.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.SPACE:
                paused = not paused
                print("paused:",paused)
    if not paused:
        if solver_type == "XPBD":
            step_xpbd(max_iter)
        elif solver_type == "GS":
            substep_all_solver(max_iter=max_iter, solver_type="GS")
        elif solver_type == "AMG":
            substep_all_solver(max_iter=max_iter, solver_type="AMG", R=R, P=P)
        if export_obj:
            write_obj(out_dir + f"{frame_num:04d}.obj", pos.to_numpy(), tri.to_numpy())
    
    if frame_num == end_frame:
        print(f"Time all: {(time.perf_counter() - timer_all):.0f}s")
        exit()
    if use_viewer:
        viewer.camera.track_user_inputs(viewer.window, movement_speed=0.003, hold_key=ti.ui.RMB)
        viewer.scene.set_camera(viewer.camera)
        viewer.scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
        viewer.scene.mesh(pos, tri, color=(1.0,0,0), two_sided=True)
        viewer.canvas.scene(viewer.scene)
        # you must call this function, even if we just want to save the image, otherwise the GUI image will not update.
        viewer.window.show()
        if save_image:
            file_path = out_dir + f"{frame_num:04d}.png"
            viewer.window.save_image(file_path)  # export and show in GUI