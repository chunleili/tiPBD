'''
在三维空间的二维绳子（z方向固定为0），沿着x轴摆放。初始拉伸1.5倍，然后释放。
'''

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


prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("prj_dir", prj_dir)
misc_dir = prj_dir + "/data/misc/"

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=100)
N = parser.parse_args().N
print("N: ", N)

frame_num = 0
end_frame = 1000
out_dir = f"./result/test/"
proj_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("proj_dir_path: ", proj_dir_path)
misc_dir_path = proj_dir_path + "/data/misc/"

save_image = True
max_iter = 50
paused = False
save_P, load_P = False, False
use_viewer = False
export_results = True
export_residual = False
solver_type = "GS" # "AMG", "GS", "XPBD"
export_matrix = True
stop_frame = 100

ti.init(arch=ti.cpu)

# N = 2
NV = N + 1
NE = N
h = 0.01
M = NE
new_M = int(NE / 100)
compliance = 1.0e-8  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h)  # timestep related compliance, see XPBD paper
omega = 0.5

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
adjacent_edge_abc   = ti.field(shape=(NE,42),  dtype = ti.int32)
adjacent_edge   = ti.field(shape=(NE,14),  dtype = ti.int32)


@ti.kernel
def init_pos(
    inv_mass:ti.template(),
    pos:ti.template(),
):
    for i in range(N+1):
        pos[i] = ti.Vector([i/N, 0, 0],ti.f32)
    for i in inv_mass:
        inv_mass[i] = 1.0
    # inv_mass[N] = 0.0
    # inv_mass[NV-1] = 0.0



@ti.kernel
def init_edge(
    edge:ti.template(),
    rest_len:ti.template(),
    pos:ti.template(),
):
    for i in range(N):
        edge[i] = ti.Vector([i, i + 1], ti.i32)
    
    for i  in range(NE):
        rest_len[i] = (pos[i] -  pos[i+1]).norm()

def init_scale():
    pos_ = pos.to_numpy()
    pos_ *= 1.5
    pos.from_numpy(pos_)

def init_random_vel():
    vel_ = vel.to_numpy()
    vel_ += np.random.rand(NV,3) * 0.1
    vel.from_numpy(vel_)

@ti.kernel
def semi_euler(
    old_pos:ti.template(),
    inv_mass:ti.template(),
    vel:ti.template(),
    pos:ti.template(),
):
    gravity = ti.Vector([0.0, 0.0, 0.0])
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
def collision(pos:ti.template()):
    for i in range(NV):
        if pos[i][2] < -2.0:
            pos[i][2] = 0.0

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
        # solve_subspace_constraints_xpbd(labels, numerator, denominator, numerator_lumped, denominator_lumped, y_jprime, dLambda, inv_mass, edge, rest_len, lagrangian, dpos, pos)
        solve_constraints_xpbd(dual_residual, inv_mass, edge, rest_len, lagrangian, dpos, pos)
        update_pos(inv_mass, dpos, pos)
        collision(pos)

        residual[i+1] = np.linalg.norm(dual_residual.to_numpy())
    np.savetxt(out_dir + f"dual_residual_{frame_num}.txt",residual)

    update_vel(old_pos, inv_mass, vel, pos)



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
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient


def transfer_back_to_pos_mfree(x):
    dLambda.from_numpy(x)
    reset_dpos(dpos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(inv_mass, dpos, pos)
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
            scipy.io.mmwrite(out_dir + f"A_f{frame_num}.mtx", A)
            np.savetxt(out_dir + f"b_f{frame_num}.txt", b)
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

def write_points_ply(filename, pos):
    cells = []
    mesh = meshio.Mesh(pos,cells)
    mesh.write(filename,file_format="ply",binary=False)


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


timer_all = time.perf_counter()
mkdir_if_not_exist(out_dir)
clean_result_dir(out_dir)
init_pos(inv_mass,pos)
init_edge(edge, rest_len, pos)
write_points_ply(prj_dir+"/result/test/0.ply", pos.to_numpy())
init_scale()
init_random_vel()

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
        if export_results:
            write_points_ply(out_dir + f"{frame_num:d}.ply", pos.to_numpy())
    
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

# if __name__ == "__main__":
#     main()