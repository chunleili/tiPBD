import taichi as ti
import numpy as np
import time
import scipy
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
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
dpos     = ti.Vector.field(3, dtype=float, shape=(NV))
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
dual_residual       = ti.field(shape=(NE),    dtype = ti.float32) # -C - alpha * lagrangian
adjacent_edge = ti.field(dtype=int, shape=(NE, 20))
num_adjacent_edge = ti.field(dtype=int, shape=(NE))
adjacent_edge_abc = ti.field(dtype=int, shape=(NE, 100))
num_nonz = 0
nnz_each_row = np.zeros(NE, dtype=int)

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
    if not scale_instead_of_attach:
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

# for debug, please do not use in real simulation. Use init_adjacent_edge_kernel instead
def init_adjacent_edge_abc():
    adjacent_edge_abc = []
    for i in range(NE):
        ii0 = edge[i][0]
        ii1 = edge[i][1]

        adj = adjacent_edge.to_numpy()[i]
        num_adj = num_adjacent_edge.to_numpy()[i]
        abc = []
        for j in range(num_adj):
            ia = adj[j]
            if ia == i:
                continue
            jj0 = edge[ia][0]
            jj1 = edge[ia][1]

            a, b, c = -1, -1, -1
            if ii0 == jj0:
                a, b, c = ii0, ii1, jj1
            elif ii0 == jj1:
                a, b, c = ii0, ii1, jj0
            elif ii1 == jj0:
                a, b, c = ii1, ii0, jj1
            elif ii1 == jj1:
                a, b, c = ii1, ii0, jj0
            abc.append(a)
            abc.append(b)
            abc.append(c)
        adjacent_edge_abc.append(abc)
    return adjacent_edge_abc

@ti.kernel
def init_adjacent_edge_abc_kernel():
    for i in range(NE):
        ii0 = edge[i][0]
        ii1 = edge[i][1]

        num_adj = num_adjacent_edge[i]
        for j in range(num_adj):
            ia = adjacent_edge[i,j]
            if ia == i:
                continue
            jj0,jj1 = edge[ia]
            a, b, c = -1, -1, -1
            if ii0 == jj0:
                a, b, c = ii0, ii1, jj1
            elif ii0 == jj1:
                a, b, c = ii0, ii1, jj0
            elif ii1 == jj0:
                a, b, c = ii1, ii0, jj1
            elif ii1 == jj1:
                a, b, c = ii1, ii0, jj0
            adjacent_edge_abc[i, j*3] = a
            adjacent_edge_abc[i, j*3+1] = b
            adjacent_edge_abc[i, j*3+2] = c


@ti.kernel
def init_adjacent_edge_kernel(adjacent_edge:ti.template(), num_adjacent_edge:ti.template(), edge:ti.template()):
    for i in range(NE):
        for j in range(adjacent_edge.shape[1]):
            adjacent_edge[i,j] = -1

    ti.loop_config(serialize=True)
    for i in range(NE):
        a=edge[i][0]
        b=edge[i][1]
        for j in range(i+1, NE):
            if j==i:
                continue
            a1=edge[j][0]
            b1=edge[j][1]
            if a==a1 or a==b1 or b==a1 or b==b1:
                numi = num_adjacent_edge[i]
                numj = num_adjacent_edge[j]
                adjacent_edge[i,numi]=j
                adjacent_edge[j,numj]=i
                num_adjacent_edge[i]+=1
                num_adjacent_edge[j]+=1 

def init_adjacent_edge(adjacent_edge, num_adjacent_edge, edge):
    for i in range(NE):
        a=edge[i][0]
        b=edge[i][1]
        for j in range(i+1, NE):
            if j==i:
                continue
            a1=edge[j][0]
            b1=edge[j][1]
            if a==a1 or a==b1 or b==a1 or b==b1:
                adjacent_edge[i][num_adjacent_edge[i]]=j
                adjacent_edge[j][num_adjacent_edge[j]]=i
                num_adjacent_edge[i]+=1
                num_adjacent_edge[j]+=1

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
    dpos:ti.template(),
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
            dpos[idx0] += invM0 * l * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * l * gradient



@ti.kernel
def solve_constraints_xpbd(
    dual_residual: ti.template(),
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    dpos:ti.template(),
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
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient

@ti.kernel
def update_pos(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]

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
def reset_dpos(dpos:ti.template()):
    for i in range(NV):
        dpos[i] = ti.Vector([0.0, 0.0, 0.0])



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
        reset_dpos(dpos)
        solve_constraints_xpbd(dual_residual, inv_mass, edge, rest_len, lagrangian, dpos, pos)
        update_pos(inv_mass, dpos, pos)

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
    vv:ti.types.ndarray(dtype=ti.f32),
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


def gauss_seidel(A, x, b, iterations=1):
    if not scipy.sparse.isspmatrix_csr(A):
        raise ValueError("A must be csr matrix!")

    for _iter in range(iterations):
        # forward sweep
        # print("forward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(A.indptr, A.indices, A.data, x, b, row_start=0, row_stop=int(len(x)), row_step=1)

        # backward sweep
        # print("backward sweeping")
        for _ in range(iterations):
            amg_core_gauss_seidel(
                A.indptr, A.indices, A.data, x, b, row_start=int(len(x)) - 1, row_stop=-1, row_step=-1
            )
    return x

def solve_amg(A, b, x0, R, P, residuals=[]):
    tol = 1e-3
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
        gauss_seidel(A, x, b, iterations=1)  # presmoother
        residual = b - A @ x
        coarse_b = R @ residual  # restriction
        coarse_x = np.zeros_like(coarse_b)
        coarse_x[:] = scipy.sparse.linalg.spsolve(A2, coarse_b)
        x += P @ coarse_x 
        gauss_seidel(A, x, b, iterations=1)
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
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient


def transfer_back_to_pos_mfree(x):
    dLambda.from_numpy(x)
    reset_dpos(dpos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(inv_mass, dpos, pos)

def spy_A(A,b):
    print("A:", A.shape, " b:", b.shape)
    scipy.io.mmwrite("A.mtx", A)
    plt.spy(A, markersize=1)
    plt.show()
    exit()

# legacy
def fill_A_by_spmm(M_inv, ALPHA):
    G_ii, G_jj, G_vv = np.zeros(M*6, dtype=np.int32), np.zeros(M*6, dtype=np.int32), np.zeros(M*6, dtype=np.float32)
    compute_C_and_gradC_kernel(pos, gradC, edge, constraints, rest_len)
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, edge)
    G = scipy.sparse.csr_array((G_vv, (G_ii, G_jj)), shape=(M, 3 * NV))
    A = G @ M_inv @ G.transpose() + ALPHA
    A = scipy.sparse.csr_matrix(A)
    return A


# fill A by directly assign value， for debug
def fill_A():
    # fill diagonal
    diags = np.zeros(NE)
    for i in range(NE):
        diags[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha
    A_diag = scipy.sparse.diags(diags)
    # A_diag = A_diag.todense()

    # fill off-diagonal
    ii, jj, vv = np.zeros(NE*NE,int), np.zeros(NE*NE,int), np.zeros(NE*NE)
    cnt_nonz = 0
    for i in range(NE):
        for j in range(num_adjacent_edge[i]):
            ia = adjacent_edge[i,j]
            a = adjacent_edge_abc[i, j * 3]
            b = adjacent_edge_abc[i, j * 3 + 1]
            c = adjacent_edge_abc[i, j * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            if offdiag == 0:
                continue
            ii[cnt_nonz] = i
            jj[cnt_nonz] = ia
            vv[cnt_nonz] = offdiag
            cnt_nonz += 1
    A_offdiag = scipy.sparse.csr_matrix((vv, (ii, jj)), shape=(NE, NE))
    # A_offdiag = A_offdiag.todense()
    A = A_diag + A_offdiag
    A = A.tocsr()
    return A


def calc_num_nonz():
    global num_nonz
    num_adj = num_adjacent_edge.to_numpy()
    num_nonz = np.sum(num_adj)+NE
    return num_nonz

def calc_nnz_each_row():
    global nnz_each_row
    num_adj = num_adjacent_edge.to_numpy()
    nnz_each_row = num_adj[:] + 1
    return nnz_each_row

def init_A_CSR_pattern():
    num_adj = num_adjacent_edge.to_numpy()
    adj = adjacent_edge.to_numpy()
    nonz = np.sum(num_adj)+NE
    indptr = np.zeros(NE+1, dtype=np.int32)
    indices = np.zeros(nonz, dtype=np.int32)
    data = np.zeros(nonz, dtype=np.float32)

    indptr[0] = 0
    for i in range(0,NE):
        num_adj_i = num_adj[i]
        indptr[i+1]=indptr[i] + num_adj_i + 1
        indices[indptr[i]:indptr[i+1]-1]= adj[i][:num_adj_i]
        indices[indptr[i+1]-1]=i

    assert indptr[-1] == nonz

    return data, indices, indptr


def csr_index_to_coo_index(indptr, indices):
    ii, jj = np.zeros_like(indices), np.zeros_like(indices)
    for i in range(NE):
        ii[indptr[i]:indptr[i+1]]=i
        jj[indptr[i]:indptr[i+1]]=indices[indptr[i]:indptr[i+1]]
    return ii, jj

# TODO:not tested
# for cnt version
def fill_A_offdiag_CSR(data, indptr, ii,jj):
    for cnt in range(num_nonz):
        i = ii[cnt] # row index
        j = jj[cnt] # col index
        k = cnt - indptr[i] #k-th non-zero element of i-th row. 
        # Because the diag is the final element of each row, 
        # it is also the k-th adjacent edge of i-th edge.
        if i == j: # diag
            continue
        a = adjacent_edge_abc[i, k * 3]
        b = adjacent_edge_abc[i, k * 3 + 1]
        c = adjacent_edge_abc[i, k * 3 + 2]
        g_ab = (pos[a] - pos[b]).normalized()
        g_ac = (pos[a] - pos[c]).normalized()
        offdiag = inv_mass[a] * g_ab.dot(g_ac)
        data[cnt] = offdiag

# for cnt version
@ti.kernel
def fill_A_offdiag_CSR_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                              indptr:ti.types.ndarray(dtype=ti.i32), 
                              ii:ti.types.ndarray(dtype=ti.i32), 
                              jj:ti.types.ndarray(dtype=ti.i32),):
    for cnt in range(num_nonz):
        i = ii[cnt] # row index
        j = jj[cnt] # col index
        k = cnt - indptr[i] #k-th non-zero element of i-th row. 
        # Because the diag is the final element of each row, 
        # it is also the k-th adjacent edge of i-th edge.
        if i == j: # diag
            continue
        a = adjacent_edge_abc[i, k * 3]
        b = adjacent_edge_abc[i, k * 3 + 1]
        c = adjacent_edge_abc[i, k * 3 + 2]
        g_ab = (pos[a] - pos[b]).normalized()
        g_ac = (pos[a] - pos[c]).normalized()
        offdiag = inv_mass[a] * g_ab.dot(g_ac)
        data[cnt] = offdiag


# For i and for k version
# Input is already in CSR format. We only update the data.
def fill_A_offdiag_CSR_2(data):
    cnt = 0
    for i in range(NE):
        for k in range(num_adjacent_edge[i]):
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            data[cnt] = offdiag
            cnt += 1
        cnt += 1 # diag


# For i and for k version
# Input is already in CSR format. We only update the data.
@ti.kernel
def fill_A_offdiag_CSR_2_kernel(data:ti.types.ndarray(dtype=ti.f32)):
    cnt = 0
    ti.loop_config(serialize=True)
    for i in range(NE):
        for k in range(num_adjacent_edge[i]):
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            data[cnt] = offdiag
            cnt += 1
        cnt += 1 # diag


# give two edge number, return the shared vertex.
def get_shared_vertex(edge1:int, edge2:int):
    a, b = edge[edge1]
    c, d = edge[edge2]
    if a == c or a == d:
        return a
    if b == c or b == d:
        return b
    return -1 # no shared vertex


def fill_A_ti():
    # fill diagonal
    # tic = time.time()
    diags = np.zeros(NE, np.float32)
    fill_A_diag_kernel(diags)
    A_diag = scipy.sparse.diags(diags)
    A_diag = A_diag.tocsr()
    # print(f"fill_A_diag time: {time.time()-tic:.3f}s")

    # fill off-diagonal
    tic = time.time()
    OFF_ii, OFF_jj, OFF_vv = np.zeros(num_nonz, int), np.zeros(num_nonz, int), np.zeros(num_nonz, np.float32)
    fill_A_offdiag_ijv_kernel(OFF_ii, OFF_jj, OFF_vv)
    A_offdiag = scipy.sparse.coo_array((OFF_vv, (OFF_ii, OFF_jj)), shape=(NE, NE))
    print(f"fill_A_offdiag_ijv_kernel time: {time.time()-tic:.3f}s")

    # # csr version fill_A_offdiag. 
    # # It is suprise that csr version is slower than ijv version.
    # # (N=64, ijv:0.074s, csr1:0.183s, csr2:0.173s)
    # # I dont know why. But let us keep the csr version for future use.
    # tic = time.time()
    # # fill_A_offdiag_CSR_kernel(data, indptr, coo_ii, coo_jj)
    # fill_A_offdiag_CSR_2_kernel(data)
    # A_offdiag_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=(NE, NE))
    # print(f"fill_A_offdiag_CSR  time: {time.time()-tic:.3f}s")
    # diff = A_offdiag.tocsr() - A_offdiag_csr
    # maxdiff = np.max(np.abs(diff.toarray()))
    # assert maxdiff < 1e-6, f"maxdiff: {maxdiff}"
    
    tic = time.time()
    A_offdiag = set_positive_offdiag_and_symmetry_to_zero_brutal(A_offdiag)
    print(f"set_positive_offdiag_and_symmetry_to_zero_brutal time: {time.time()-tic:.3f}s")
    tic = time.time()
    A_offdiag1 = set_positive_offdiag_and_symmetry_to_zero_ti(A_offdiag)
    print(f"set_positive_offdiag_and_symmetry_to_zero_ti time: {time.time()-tic:.3f}s")

    maxdiff = np.max(np.abs(A_offdiag.toarray()-A_offdiag1.toarray()))
    assert maxdiff < 1e-6, f"maxdiff: {maxdiff}"
    print(f"maxdiff: {maxdiff}")

    mmwrite("A_offdiag.mtx", A_offdiag)
    # tic = time.time()
    A_offdiag = A_offdiag.tocsr()
    A = A_diag + A_offdiag
    A = A.tocsr()
    # print(f"fill_A plus time: {time.time()-tic:.3f}s")
    return A


@ti.kernel
def fill_A_diag_kernel(diags:ti.types.ndarray(dtype=ti.f32)):
    for i in range(NE):
        diags[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha


@ti.kernel
def fill_A_offdiag_ijv_kernel(ii:ti.types.ndarray(dtype=ti.i32), jj:ti.types.ndarray(dtype=ti.i32), vv:ti.types.ndarray(dtype=ti.f32)):
    cnt_nonz = 0
    ti.loop_config(serialize=True)
    for i in range(NE):
        for j in range(num_adjacent_edge[i]):
            ia = adjacent_edge[i,j]
            a = adjacent_edge_abc[i, j * 3]
            b = adjacent_edge_abc[i, j * 3 + 1]
            c = adjacent_edge_abc[i, j * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            # if offdiag == 0:
            #     continue
            ii[cnt_nonz] = i
            jj[cnt_nonz] = ia
            vv[cnt_nonz] = offdiag
            cnt_nonz += 1
        cnt_nonz += 1 # diag placeholder



@ti.kernel
def fill_A_diag_and_offdiag_kernel(ii:ti.types.ndarray(dtype=ti.i32), jj:ti.types.ndarray(dtype=ti.i32), vv:ti.types.ndarray(dtype=ti.f32)):
    cnt_nonz = 0

    for i in range(NE):
        vv[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha
        ii[cnt_nonz] = i
        jj[cnt_nonz] = i
        cnt_nonz += 1

    ti.loop_config(serialize=True)
    for i in range(NE):
        for j in range(num_adjacent_edge[i]):
            ia = adjacent_edge[i,j]
            a = adjacent_edge_abc[i, j * 3]
            b = adjacent_edge_abc[i, j * 3 + 1]
            c = adjacent_edge_abc[i, j * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            if offdiag == 0:
                continue
            ii[cnt_nonz] = i
            jj[cnt_nonz] = ia
            vv[cnt_nonz] = offdiag
            cnt_nonz += 1

def set_positive_offdiag_and_symmetry_to_zero_brutal(A):
    A = A.tocsr()
    for i in range(A.shape[0]):
        for k in range(A[[i]].nnz):
            j = A.indices[A.indptr[i]+k]
            if A[i,j] > 0:
                A[i,j] = 0
                A[j,i] = 0
    return A


def set_positive_offdiag_and_symmetry_to_zero_ti(A):
    A = A.tocsr()
    nrow = A.shape[0]
    global nnz_each_row
    indices = A.indices
    data = A.data
    indptr = A.indptr
    set_positive_offdiag_and_symmetry_to_zero_kernel(nrow, nnz_each_row, indices, indptr,data)
    return A

@ti.kernel
def set_positive_offdiag_and_symmetry_to_zero_kernel(
    nrow:ti.i32,
    nnz_each_row:ti.types.ndarray(dtype=ti.i32),
    indices:ti.types.ndarray(dtype=ti.i32),
    indptr:ti.types.ndarray(dtype=ti.i32),
    data:ti.types.ndarray(dtype=ti.f32),
):
    for i in range(nrow):
        for k in range(nnz_each_row[i]):
            j = indices[indptr[i]+k]
            if data[indptr[i]+k] > 0:
                data[indptr[i]+k] = 0
            if data[indptr[j]+k] != 0:
                data[indptr[j]+k] = 0

# # #  with bug
# def set_positive_offdiag_and_symmetry_to_zero(A_offdiag):
#     data = A_offdiag.data
#     cnt = np.where(data > 0) # index of positive off-diagonal elements
#     symmetry=(A_offdiag == A_offdiag.T).toarray().all()

#     A_offdiag = A_offdiag.tocsr()
#     indptr =  A_offdiag.tocsr().indptr

#     A_offdiag = A_offdiag.tocoo()
#     ii = A_offdiag.tocoo().row
#     jj = A_offdiag.tocoo().col
#     i = ii[cnt] # row index of positive off-diagonal elements
#     j = jj[cnt] # col index of positive off-diagonal elements
#     cnt2 = ij_to_cnt(i, j, indptr) # symmetric index of positive off-diagonal elements. BUG: cnt2 may not in the A.

#     data[cnt] *= 1e-3
#     # data[cnt2] *= 1e-3

#     A_offdiag.data = data
#     return A_offdiag



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
        tic = time.time()
        # A1 = fill_A_by_spmm(M_inv, ALPHA)
        A = fill_A_ti()
        print(f"fill_A time: {time.time()-tic:.3f}s")

        # maxdiff = np.max(np.abs(A1.toarray()-A.toarray()))
        # assert maxdiff < 1e-6, f"maxdiff: {maxdiff}"
        # print(f"maxdiff: {maxdiff}")

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
            import pyamg
            # print("generating R and P by pyamg...")
            ml = pyamg.ruge_stuben_solver(A, max_levels=2)
            P = ml.levels[0].P
            R = ml.levels[0].R
            # print(f"R: {R.shape}, P: {P.shape}")
            x = solve_amg(A, b, x0, R, P, residuals=[])
        
        transfer_back_to_pos_mfree(x)

        if export_residual:
            linsys_r = np.linalg.norm(A @ x - b)
            calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
            dual_r = np.linalg.norm(dual_residual.to_numpy())
            print(f"iter {ite} r: {linsys_r:.2e}, dual_r: {dual_r:.2e}")
            # with open(out_dir+f"r_frame_{frame_num}.txt", 'a+') as f:
            #     f.write(f"{linsys_r}\n")
            # with open(out_dir+f"dual_r_frame_{frame_num}.txt", 'a+') as f:
            #     f.write(f"{dual_r}\n")
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
    from pathlib import Path
    pwd = os.getcwd()
    os.chdir(folder_path)
    print(f"clean {folder_path}...")
    except_files = ["b0.txt"]
    to_remove = []
    for wildcard_name in [
        '*.obj',
        '*.png',
        '*.ply',
        # '*.txt',
    ]:
        files = glob.glob(wildcard_name)
        to_remove += (files)
        for f in files:
            if f in except_files:
                to_remove.remove(f)
    print(f"removing {len(to_remove)} files")
    for file_path in to_remove:
        os.remove(file_path)
    print(f"clean {folder_path} done")
    os.chdir(pwd)

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

@ti.kernel
def init_scale():
    scale = 1.5
    for i in range(NV):
        pos[i] *= scale

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
scale_instead_of_attach = True
use_offdiag = True

timer_all = time.perf_counter()
init_pos(inv_mass,pos)
init_tri(tri)
init_edge(edge, rest_len, pos)
write_obj(out_dir + f"{frame_num:04d}.obj", pos.to_numpy(), tri.to_numpy())
if scale_instead_of_attach:
    init_scale()

#init adjacent edge
tic = time.time()
init_adjacent_edge_kernel(adjacent_edge, num_adjacent_edge, edge)
adjacent_edge_abc.fill(-1)
init_adjacent_edge_abc_kernel()
print(f"init_adjacent_edge and abc time: {time.time()-tic:.3f}s")

#calculate number of nonzeros by counting number of adjacent edges
num_nonz = calc_num_nonz() 
nnz_each_row = calc_nnz_each_row()

# init csr pattern. In the future we will replace all ijv pattern with csr
data, indices, indptr = init_A_CSR_pattern()
coo_ii, coo_jj = csr_index_to_coo_index(indptr, indices)

# [row,col] to 1D index for csr or coo format
def ij_to_cnt(i,j, indptr):
    return indptr[i]+j

if solver_type=="AMG":
    init_edge_center(edge_center, edge, pos)
    if save_P:
        R, P, labels, new_M = compute_R_and_P_kmeans()
        scipy.io.mmwrite(misc_dir_path + "R.mtx", R)
        scipy.io.mmwrite(misc_dir_path + "P.mtx", P)
        np.savetxt(misc_dir_path + "labels.txt", labels, fmt="%d")
    if load_P:
        R = scipy.io.mmread(misc_dir_path+ "R.mtx")
        P = scipy.io.mmread(misc_dir_path+ "P.mtx")
        # labels = np.loadtxt( "labels.txt", dtype=np.int32)

print("Initialization done. Cost time: ", time.perf_counter() - timer_all, "s")

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