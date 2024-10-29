import scipy.sparse
import taichi as ti
import numpy as np
import time
import scipy
from pathlib import Path
import os,sys
from matplotlib import pyplot as plt
import tqdm
import argparse
from collections import namedtuple
import json
import logging
import datetime
from time import perf_counter


prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(prj_path)
from engine.file_utils import process_dirs,  do_restart, save_state,  export_A_b
from engine.init_extlib import init_extlib
from engine.mesh_io import write_mesh
from engine.solver.build_Ps import build_Ps
from engine.cloth.bending import init_bending, solve_bending_constraints_xpbd
from engine.solver.amg_python import AmgPython
from engine.solver.amg_cuda import AmgCuda
from engine.solver.amgx_solver import AmgxSolver
from engine.solver.direct_solver import DirectSolver
from engine.util import is_stall, ending, is_diverge



#parse arguments to change default values
from engine.common_args import add_common_args
parser = argparse.ArgumentParser()
parser = add_common_args(parser)

parser.add_argument("-N", type=int, default=64)
parser.add_argument("-compliance", type=float, default=1.0e-8)
parser.add_argument("-use_PXPBD_v1", type=int, default=False)
parser.add_argument("-use_PXPBD_v2", type=int, default=False)
parser.add_argument("-use_bending", type=int, default=False)
parser.add_argument("-setup_num", type=int, default=0, help="attach:0, scale:1")

parser.add_argument("-omega", type=float, default=0.25)
parser.add_argument("-smoother_type", type=str, default="chebyshev")

args = parser.parse_args()

N = args.N

if args.setup_num==1: args.gravity = (0.0, 0.0, 0.0)
else : args.gravity = (0.0, -9.8, 0.0)

args.save_image = True
args.use_viewer = False
args.export_fullr = False
args.calc_r_xpbd = True
args.use_cpp_initFill = True
args.PXPBD_ksi = 1.0


if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

if args.use_PXPBD_v1 or args.use_PXPBD_v2:
    ResidualDataPrimal = namedtuple('residual', ['dual','primal','Newton', 'ninner','t'])
    Residual0DataPrimal = namedtuple('residual', ['dual','primal','Newton'])
else:
    ResidualData = namedtuple('residual', ['dual', 'ninner','t'])
    Residual0Data = namedtuple('residual', ['dual'])

class Cloth():
    def __init__(self) -> None:
        self.Ps = None
        self.num_levels = 0
        self.paused = False
        self.n_outer_all = [] 
        self.all_stalled = [] 
        self.t_export = 0.0
        self.sim_name=f"cloth-N{args.N}"

        self.frame = 0
        self.ite=0
        self.NV = (N + 1)**2
        self.NT = 2 * N**2
        self.NE = 2 * N * (N + 1) + N**2
        self.NCONS = self.NE
        self.new_M = int(self.NE / 100)
        self.compliance = args.compliance  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
        self.alpha = self.compliance * (1.0 / args.delta_t / args.delta_t)  # timestep related compliance, see XPBD self.paper
        self.alpha_bending = 1.0 * (1.0 / args.delta_t / args.delta_t) #TODO: need to be tuned
    
        self.ALPHA = None
        
        self.tri = ti.field(ti.i32, shape=3 * self.NT)
        self.edge        = ti.Vector.field(2, dtype=int, shape=(self.NE))
        self.pos         = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.dpos        = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.dpos_withg  = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.old_pos     = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.vel         = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.pos_mid     = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.inv_mass    = ti.field(dtype=float, shape=(self.NV))
        self.rest_len    = ti.field(dtype=float, shape=(self.NE))
        self.lagrangian  = ti.field(dtype=float, shape=(self.NE))  
        self.constraints = ti.field(dtype=float, shape=(self.NE))  
        self.dLambda     = ti.field(dtype=float, shape=(self.NE))
        # self.# numerator   = ti.field(dtype=float, shape=(self.NE))
        # self.# denominator = ti.field(dtype=float, shape=(self.NE))
        self.gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(self.NE,2)) 
        self.edge_center = ti.Vector.field(3, dtype = ti.float32, shape=(self.NE))
        self.dual_residual       = ti.field(shape=(self.NE),    dtype = ti.float32) # -C - alpha * self.lagrangian
        self.nnz_each_row = np.zeros(self.NE, dtype=int)
        self.potential_energy = ti.field(dtype=float, shape=())
        self.inertial_energy = ti.field(dtype=float, shape=())
        self.predict_pos = ti.Vector.field(3, dtype=float, shape=(self.NV))
        # self.# primary_residual = np.zeros(dtype=float, shape=(3*self.NV))
        # self.# K = ti.Matrix.field(3, 3, float, (self.NV, self.NV)) 
        # self.# geometric stiffness, only retain diagonal elements
        self.K_diag = np.zeros((self.NV*3), dtype=float)
        # self.# Minv_gg = ti.Vector.field(3, dtype=float, shape=(self.NV))


@ti.kernel
def init_pos(
    inv_mass:ti.template(),
    pos:ti.template(),
    N:ti.i32,
    NV:ti.i32,
):
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        # pos[idx] = ti.Vector([i / N,  j / N, 0.5])  # vertical hang
        pos[idx] = ti.Vector([i / N, 0.5, j / N]) # horizontal hang
        inv_mass[idx] = 1.0
    if args.setup_num == 0:
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
    for i in range(ist.NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()



@ti.kernel
def semi_euler(
    old_pos:ti.template(),
    inv_mass:ti.template(),
    vel:ti.template(),
    pos:ti.template(),
    predict_pos:ti.template(),
    delta_t:ti.f32,
):
    g = ti.Vector(args.gravity)
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            vel[i] += delta_t * g
            old_pos[i] = pos[i]
            pos[i] += delta_t * vel[i]
            predict_pos[i] = pos[i]



@ti.kernel
def solve_distance_constraints_xpbd(
    dual_residual: ti.template(),
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian[i] * ist.alpha) / (invM0 + invM1 + ist.alpha)
        lagrangian[i] += delta_lagrangian

        # residual
        dual_residual[i] = -(constraint + ist.alpha * lagrangian[i])
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient



@ti.kernel
def update_pos(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    omega:ti.f32,
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]


@ti.kernel
def update_pos_blend(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    dpos_withg:ti.template(),
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            pos[i] += args.omega *((1-args.PXPBD_ksi) * dpos[i] + args.PXPBD_ksi * dpos_withg[i])


@ti.kernel
def update_vel(
    old_pos:ti.template(),
    inv_mass:ti.template(),    
    vel:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / args.delta_t


@ti.kernel 
def reset_dpos(dpos:ti.template()):
    for i in range(ist.NV):
        dpos[i] = ti.Vector([0.0, 0.0, 0.0])



@ti.kernel
def calc_dual_residual(
    dual_residual: ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]

        # residual(lagrangian=0 for first iteration)
        dual_residual[i] = -(constraint + ist.alpha * lagrangian[i])

def calc_primary_residual(G,M_inv):
    MASS = scipy.sparse.diags(1.0/(M_inv.diagonal()+1e-12), format="csr")
    primary_residual = MASS @ (ist.pos.to_numpy().flatten() - ist.predict_pos.to_numpy().flatten()) - G.transpose() @ ist.lagrangian.to_numpy()
    where_zeros = np.where(M_inv.diagonal()==0)
    primary_residual = np.delete(primary_residual, where_zeros)
    return primary_residual


def xpbd_calcr(tic_iter, dual0, r):
    tic_calcr = perf_counter()
    t_iter = perf_counter()-tic_iter
    # dualr = np.linalg.norm(dual_residual.to_numpy())
    dualr = calc_norm(ist.dual_residual)
    
    # if export_fullr:
    #     np.savez(args.out_dir+'/r/'+ f'fulldual_{ist.frame}-{ist.ite}', fulldual0)

    t_calcr = perf_counter()-tic_calcr
    tic_exportr = perf_counter()
    r.append(ResidualData(dualr, 1, t_iter))
    if args.export_log:
        logging.info(f"{ist.frame}-{ist.ite}  dual0:{dual0:.2e} dual:{dualr:.2e}  t:{t_iter:.2e}s calcr:{t_calcr:.2e}s")
    ist.t_export += perf_counter() - tic_exportr
    return dualr, dual0


@ti.kernel
def calc_norm(a:ti.template())->ti.f32:
    sum = 0.0
    for i in range(a.shape[0]):
        sum += a[i] * a[i]
    sum = ti.sqrt(sum)
    return sum




def substep_xpbd():
    semi_euler(ist.old_pos, ist.inv_mass, ist.vel, ist.pos, ist.predict_pos,args.delta_t)
    reset_lagrangian(ist.lagrangian)

    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    fulldual0 = ist.dual_residual.to_numpy()
    dual0 = np.linalg.norm(fulldual0).astype(float)
    r = []
    for ist.ite in range(args.maxiter):
        tic_iter = perf_counter()

        reset_dpos(ist.dpos)
        if args.use_bending:
            # TODO: should use seperate dual_residual_bending and lagrangian_bending
            solve_bending_constraints_xpbd(ist.dual_residual, ist.inv_mass, ist.lagrangian, ist.dpos, ist.pos, ist.bending_length, ist.tri_pairs, ist.alpha_bending)
        solve_distance_constraints_xpbd(ist.dual_residual, ist.inv_mass, ist.edge, ist.rest_len, ist.lagrangian, ist.dpos, ist.pos)
        update_pos(ist.inv_mass, ist.dpos, ist.pos,args.omega)

        if args.calc_r_xpbd:
            dualr, dualr0 = xpbd_calcr(tic_iter, dual0, r)

        if dualr<args.tol:
            break
        # if is_stall(r):
        #     logging.info("Stall detected, break")
        #     break
    ist.n_outer_all.append(ist.ite+1)

    if args.export_residual:
        do_export_r(r)
    update_vel(ist.old_pos, ist.inv_mass, ist.vel, ist.pos)




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
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]
        g = dis.normalized()

        gradC[i, 0] = g
        gradC[i, 1] = -g


@ti.kernel
def compute_K_kernel(K_diag:ti.types.ndarray(),):
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        dis = ist.pos[idx0] - ist.pos[idx1]
        L= dis.norm()
        g = dis.normalized()

        #geometric stiffness K: 
        # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
        # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
        # https://blog.csdn.net/weixin_43940314/article/details/139448858
        # geometric stiffness
        """
            k = lambda[i]/l * (I - n * n')
            K = | Hessian_{x1,x1}, Hessian_{x1,x2}   |  = | k  -k|
                | Hessian_{x1,x2}, Hessian_{x2,x2}   |    |-k   k|
        """
        k0 = ist.lagrangian[i] / L * (1 - g[0]*g[0])
        k1 = ist.lagrangian[i] / L * (1 - g[1]*g[1])
        k2 = ist.lagrangian[i] / L * (1 - g[2]*g[2])
        K_diag[idx0*3]   = k0
        K_diag[idx0*3+1] = k1
        K_diag[idx0*3+2] = k2
        K_diag[idx1*3]   = k0
        K_diag[idx1*3+1] = k1
        K_diag[idx1*3+2] = k2
    ...


@ti.kernel
def update_constraints_kernel(
    pos:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    constraints:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]


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
    for i in range(ist.NE):
        lagrangian[i] = 0.0




# @ti.kernel
# def calc_chen2023_added_dpos(G, M_inv, Minv_gg, dLambda):
#     dpos = M_inv @ G.transpose() @ dLambda
#     dpos -= Minv_gg
#     return dpos

def transfer_back_to_pos_matrix(x, M_inv, G, pos_mid, Minv_gg=None):
    dLambda_ = x.copy()
    ist.lagrangian.from_numpy(ist.lagrangian.to_numpy() + dLambda_)
    dpos = M_inv @ G.transpose() @ dLambda_ 
    if args.use_PXPBD_v1:
        dpos -=  Minv_gg
    dpos = dpos.reshape(-1, 3)
    ist.pos.from_numpy(pos_mid.to_numpy() + args.omega*dpos)

@ti.kernel
def transfer_back_to_pos_mfree_kernel():
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = ist.inv_mass[idx0], ist.inv_mass[idx1]

        delta_lagrangian = ist.dLambda[i]
        ist.lagrangian[i] += delta_lagrangian

        gradient = ist.gradC[i, 0]
        
        if invM0 != 0.0:
            ist.dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            ist.dpos[idx1] -= invM1 * delta_lagrangian * gradient


@ti.kernel
def transfer_back_to_pos_mfree_kernel_withg():
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = ist.inv_mass[idx0], ist.inv_mass[idx1]
        gradient = ist.gradC[i, 0]
        if invM0 != 0.0:
            ist.dpos_withg[idx0] += invM0 * ist.lagrangian[i] * gradient 
        if invM1 != 0.0:
            ist.dpos_withg[idx1] -= invM1 * ist.lagrangian[i] * gradient

    for i in range(ist.NV):
        if ist.inv_mass[i] != 0.0:
            ist.dpos_withg[i] += ist.predict_pos[i] - ist.old_pos[i]


def transfer_back_to_pos_mfree(x):
    ist.dLambda.from_numpy(x)
    reset_dpos(ist.dpos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(ist.inv_mass, ist.dpos, ist.pos, args.omega)






@ti.kernel
def compute_potential_energy():
    ist.potential_energy[None] = 0.0
    ist.inv_alpha = 1.0/ist.compliance
    for i in range(ist.NE):
        ist.potential_energy[None] += 0.5 * ist.inv_alpha * ist.constraints[i]**2

@ti.kernel
def compute_inertial_energy():
    ist.inertial_energy[None] = 0.0
    ist.inv_h2 = 1.0 / ist.delta_t**2
    for i in range(ist.NV):
        if ist.inv_mass[i] == 0.0:
            continue
        ist.inertial_energy[None] += 0.5 / ist.inv_mass[i] * (ist.pos[i] - ist.predict_pos[i]).norm_sqr() * ist.inv_h2




def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))


def report_multilevel_details(Ps, num_levels):
    logging.info(f"    num_levels:{num_levels}")
    num_points_level = []
    for i in range(len(Ps)):
        num_points_level.append(Ps[i].shape[0])
    num_points_level.append(Ps[-1].shape[1])
    for i in range(num_levels):
        logging.info(f"    num points of level {i}: {num_points_level[i]}")


def should_setup():
    if ist.ite != 0:
        return False
    if ist.frame==1:
        return True
    if ist.frame%args.setup_interval==0:
        return True
    if args.restart and ist.frame==ist.initial_frame:
        return True
    return False



def AMG_calc_r(r, r0, tic_iter, r_Axb):
    tic = time.perf_counter()

    t_iter = perf_counter()-tic_iter
    tic_calcr = perf_counter()
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    dual_r = np.linalg.norm(ist.dual_residual.to_numpy()).astype(float)
    # compute_potential_energy()
    # compute_inertial_energy()
    # robj = (potential_energy[None]+inertial_energy[None])
    r_Axb = r_Axb.tolist() if not isinstance(r_Axb, list) else r_Axb
    if args.use_PXPBD_v1 or args.use_PXPBD_v2:
        G = fill_G()
        primary_residual = calc_primary_residual(G, ist.M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.linalg.norm(np.concatenate((ist.dual_residual.to_numpy(), primary_residual))).astype(float)

    # if export_fullr:
    #     fulldual_final = dual_residual.to_numpy()
    #     np.savez_compressed(args.out_dir+'/r/'+ f'fulldual_{ist.frame}-{ist.ite}', fulldual0, fulldual_final)

    logging.info(f"    convergence factor: {calc_conv(r_Axb):.2f}")
    logging.info(f"    Calc r time: {(perf_counter()-tic_calcr)*1000:.0f}ms")

    if args.export_log:
        logging.info(f"    iter total time: {t_iter*1000:.0f}ms")
        if args.use_PXPBD_v1 or args.use_PXPBD_v2:
            r.append(ResidualDataPrimal(dual_r, primal_r, Newton_r, len(r_Axb), t_iter))
            logging.info(f"{ist.frame}-{ist.ite} Newton:{Newton_r:.2e} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual:{dual_r:.2e} primal:{primal_r:.2e} iter:{len(r_Axb)}")
        else:
            r.append(ResidualData(dual_r, len(r_Axb), t_iter))
            logging.info(f"{ist.frame}-{ist.ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual0:{r0.dual:.2e} dual:{dual_r:.2e}  iter:{len(r_Axb)}")

    ist.t_export += perf_counter()-tic



def AMG_calc_r0():
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    dual_r = np.linalg.norm(ist.dual_residual.to_numpy()).astype(float)
    # compute_potential_energy()
    # compute_inertial_energy()
    # robj = (potential_energy[None]+inertial_energy[None])
    if args.use_PXPBD_v1 or args.use_PXPBD_v2:
        G = fill_G()
        primary_residual = calc_primary_residual(G, ist.M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.linalg.norm(np.concatenate((ist.dual_residual.to_numpy(), primary_residual))).astype(float)

        r0 = (Residual0DataPrimal(dual_r, primal_r, Newton_r))
    else:
        r0 = (Residual0Data(dual_r))
    return r0


def do_export_r(r):
    tic = time.perf_counter()
    serialized_r = [r[i]._asdict() for i in range(len(r))]
    r_json = json.dumps(serialized_r)
    with open(args.out_dir+'/r/'+ f'{ist.frame}.json', 'w') as file:
        file.write(r_json)
    ist.t_export += time.perf_counter()-tic


@ti.kernel
def PXPBD_b_kernel(pos:ti.template(), predict_pos:ti.template(), lagrangian:ti.template(), inv_mass:ti.template(), gradC:ti.template(), b:ti.types.ndarray(), Minv_gg:ti.template()):
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        if invM0 != 0.0:
            Minv_gg[idx0] = invM0 * lagrangian[i] * gradC[i, 0] + (pos[idx0] - predict_pos[idx0])
        if invM1 != 0.0:
            Minv_gg[idx1] = invM1 * lagrangian[i] * gradC[i, 1] + (pos[idx0] - predict_pos[idx0])

    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        if invM1 != 0.0 and invM0 != 0.0:
            b[idx0] += gradC[i, 0] @ Minv_gg[idx0] + gradC[i, 1] @ Minv_gg[idx1]

        #     Minv_gg =  (pos.to_numpy().flatten() - predict_pos.to_numpy().flatten()) - M_inv @ G.transpose() @ lagrangian.to_numpy()
        #     b += G @ Minv_gg


# v1-mfree
def PXPBD_v1_mfree_transfer_back_to_pos(x, Minv_gg):
    ist.dLambda.from_numpy(x)
    reset_dpos(ist.dpos)
    PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg)
    update_pos(ist.inv_mass, ist.dpos, ist.pos)


# v1-mfree
@ti.kernel
def PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg:ti.template()):
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = ist.inv_mass[idx0], ist.inv_mass[idx1]

        delta_lagrangian = ist.dLambda[i]
        ist.lagrangian[i] += delta_lagrangian

        gradient = ist.gradC[i, 0]
        
        if invM0 != 0.0:
            ist.dpos[idx0] += invM0 * delta_lagrangian * gradient - Minv_gg[idx0]
        if invM1 != 0.0:
            ist.dpos[idx1] -= invM1 * delta_lagrangian * gradient - Minv_gg[idx1]
        


# original XPBD dlam2dpos
def AMG_dlam2dpos(x):
    tic = time.perf_counter()
    transfer_back_to_pos_mfree(x)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")


# v1: with g, modify b and dpos
def AMG_PXPBD_v1_dlam2dpos(x,G, Minv_gg):
    dLambda_ = x.copy()
    ist.lagrangian.from_numpy(ist.lagrangian.to_numpy() + dLambda_)
    dpos = ist.M_inv @ G.transpose() @ dLambda_ 
    dpos -=  Minv_gg
    dpos = dpos.reshape(-1, 3)
    ist.pos.from_numpy(ist.pos_mid.to_numpy() + ist.omega*dpos)


# v2: blended, only modify dpos
def AMG_PXPBD_v2_dlam2dpos(x):
    tic = time.perf_counter()
    ist.dLambda.from_numpy(x)
    reset_dpos(ist.dpos)
    ist.dpos_withg.fill(0)
    transfer_back_to_pos_mfree_kernel()
    update_pos(ist.inv_mass, ist.dpos, ist.pos)
    compute_C_and_gradC_kernel(ist.pos, ist.gradC, ist.edge, ist.constraints, ist.rest_len) # required by dlam2dpos
    # G = fill_G()
    transfer_back_to_pos_mfree_kernel_withg()
    # dpos_withg_np = (predict_pos.to_numpy() - pos.to_numpy()).flatten() + M_inv @ G.transpose() @ lagrangian.to_numpy()
    # dpos_withg.from_numpy(dpos_withg_np.reshape(-1, 3))
    update_pos_blend(ist.inv_mass, ist.dpos, ist.pos, ist.dpos_withg)
    update_pos(ist.inv_mass, ist.dpos_withg, ist.pos)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")


def AMG_b():
    update_constraints_kernel(ist.pos, ist.edge, ist.rest_len, ist.constraints)
    b = -ist.constraints.to_numpy() - ist.alpha_tilde_np * ist.lagrangian.to_numpy()
    return b


def AMG_PXPBD_v1_b(G):
    # #we calc inverse mass times gg(primary residual), because NCONS may contains infinity for fixed pin points. And gg always appears with inv_mass.
    update_constraints_kernel(ist.pos, ist.edge, ist.rest_len, ist.constraints)
    b = -ist.constraints.to_numpy() - ist.alpha_tilde_np * ist.lagrangian.to_numpy()

    # PXPBD_b_kernel(pos, predict_pos, lagrangian, inv_mass, gradC, b, Minv_gg)
    MASS = scipy.sparse.diags(1.0/(ist.M_inv.diagonal()+1e-12), format="csr")
    Minv_gg =  MASS@ist.M_inv@(ist.pos.to_numpy().flatten() - ist.predict_pos.to_numpy().flatten()) - ist.M_inv @ G.transpose() @ ist.lagrangian.to_numpy()
    b += G @ Minv_gg
    return b, Minv_gg
    



def calc_dual():
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    return ist.dual_residual.to_numpy()


def substep_all_solver():
    tic1 = time.perf_counter()
    semi_euler(ist.old_pos, ist.inv_mass, ist.vel, ist.pos, ist.predict_pos, args.delta_t)
    reset_lagrangian(ist.lagrangian)
    r = [] # residual list of one ist.frame
    # fulldual0 = calc_dual()
    # print("dual0: ", np.linalg.norm(fulldual0))
    logging.info(f"pre-loop time: {(perf_counter()-tic1)*1000:.0f}ms")
    r0 = AMG_calc_r0()
    for ist.ite in range(args.maxiter):
        tic_iter = perf_counter()
        if args.use_PXPBD_v1:
            copy_field(ist.pos_mid, ist.pos)
        compute_C_and_gradC_kernel(ist.pos, ist.gradC, ist.edge, ist.constraints, ist.rest_len) # required by dlam2dpos
        b = AMG_b()
        if args.use_PXPBD_v1:
            G = fill_G()
            b, Minv_gg = AMG_PXPBD_v1_b(G)
        x, r_Axb = amg.run(b)
        if args.use_PXPBD_v1:
            AMG_PXPBD_v1_dlam2dpos(x, G, Minv_gg)
        elif args.use_PXPBD_v2:
            AMG_PXPBD_v2_dlam2dpos(x)
        else:
            AMG_dlam2dpos(x)
        AMG_calc_r(r, r0, tic_iter, r_Axb)
        logging.info(f"iter time(with export): {(perf_counter()-tic_iter)*1000:.0f}ms")

        if is_diverge(r, r_Axb, ist):
            raise ValueError(f"Diverge detected at frame {ist.frame}, ite {ist.ite}")

        if is_stall(r, ist, args):
            logging.info("Stall detected, break")
            break

        if args.use_PXPBD_v1:
            rtol = 1e-1
            if  r[-1].Newton<rtol*r[0].Newton:
                break
        else:
            if r[-1].dual<args.tol:
                break

    
    tic = time.perf_counter()
    logging.info(f"n_outer: {ist.ite+1}")
    ist.n_outer_all.append(ist.ite+1)
    if args.export_residual:
        do_export_r(r)
    update_vel(ist.old_pos, ist.inv_mass, ist.vel, ist.pos)
    logging.info(f"post-loop time: {(time.perf_counter()-tic)*1000:.0f}ms")



@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]




@ti.kernel
def init_scale():
    scale = 1.5
    for i in range(ist.NV):
        ist.pos[i] *= scale



# ---------------------------------------------------------------------------- #
#                                 start fill A                                 #
# ---------------------------------------------------------------------------- #
def dict_to_ndarr(d:dict)->np.ndarray:
    lengths = np.array([len(v) for v in d.values()])

    max_len = max(len(item) for item in d.values())
    # 使用填充或截断的方式转换为NumPy数组
    arr = np.array([list(item) + [-1]*(max_len - len(item)) if len(item) < max_len else list(item)[:max_len] for item in d.values()])
    return arr, lengths

class FillACloth():
    def load(self):
        self.cache_and_initFill()

    def initFill_python(self):
        tic1 = perf_counter()
        print("Initializing adjacent edge and abc...")
        ist.adjacent_edge, v2e_dict = self.init_adj_edge(edges=ist.edge.to_numpy())
        ist.adjacent_edge,ist.num_adjacent_edge = dict_to_ndarr(ist.adjacent_edge)
        v2e_np, num_v2e = dict_to_ndarr(v2e_dict)

        ist.adjacent_edge_abc = np.empty((ist.NE, 20*3), dtype=np.int32)
        ist.adjacent_edge_abc.fill(-1)
        self.init_adjacent_edge_abc_kernel(ist.NE,ist.edge,ist.adjacent_edge,ist.num_adjacent_edge,ist.adjacent_edge_abc)

        ist.num_nonz = self.calc_num_nonz(ist.num_adjacent_edge) 
        data, indices, indptr = self.init_A_CSR_pattern(ist.num_adjacent_edge, ist.adjacent_edge)
        ii, jj = self.csr_index_to_coo_index(indptr, indices)
        print(f"initFill time: {perf_counter()-tic1:.3f}s")
        return ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, data, indices, indptr, ii, jj, v2e_np, num_v2e

    def initFill_cpp(self):
        tic1 = perf_counter()
        print("Initializing adjacent edge and abc...")
        extlib.initFillCloth_set(ist.edge.to_numpy(), ist.NE)
        extlib.initFillCloth_run()
        ist.num_nonz = extlib.initFillCloth_get_nnz()

        MAX_ADJ = 20
        MAX_V2E = MAX_ADJ
        ist.adjacent_edge = np.zeros((ist.NE, MAX_ADJ), dtype=np.int32)
        ist.num_adjacent_edge = np.zeros(ist.NE, dtype=np.int32)
        ist.adjacent_edge_abc = np.zeros((ist.NE, MAX_ADJ*3), dtype=np.int32)
        ist.spmat_data = np.zeros(ist.num_nonz, dtype=np.float32)
        ist.spmat_indices = np.zeros(ist.num_nonz, dtype=np.int32)
        ist.spmat_indptr = np.zeros(ist.NE+1, dtype=np.int32)
        ist.spmat_ii = np.zeros(ist.num_nonz, dtype=np.int32)
        ist.spmat_jj = np.zeros(ist.num_nonz, dtype=np.int32)
        ist.v2e = np.zeros((ist.NV, MAX_V2E), dtype=np.int32)
        ist.num_v2e = np.zeros(ist.NV, dtype=np.int32)

        extlib.initFillCloth_get(ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.v2e, ist.num_v2e)
        print(f"initFill time: {perf_counter()-tic1:.3f}s")
        return ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.v2e, ist.num_v2e

    def cache_and_initFill(self):
        if  os.path.exists(f'cache_initFill_N{N}.npz') and args.use_cache:
            npzfile= np.load(f'cache_initFill_N{N}.npz')
            (ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj) = (npzfile[key] for key in ['adjacent_edge', 'num_adjacent_edge', 'adjacent_edge_abc', 'num_nonz', 'spmat_data', 'spmat_indices', 'spmat_indptr', 'spmat_ii', 'spmat_jj'])
            ist.num_nonz = int(ist.num_nonz) # npz save int as np.array, it will cause bug in taichi kernel
            print(f"load cache_initFill_N{N}.npz")
        else:
            if args.use_cuda and args.use_cpp_initFill:
                initFill = self.initFill_cpp
            else:
                initFill = self.initFill_python
            ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.v2e, ist.num_v2e = initFill()
            print("caching init fill...")
            tic = perf_counter() # savez_compressed will save 10x space(1.4G->140MB), but much slower(33s)
            np.savez(f'cache_initFill_N{N}.npz', adjacent_edge=ist.adjacent_edge, num_adjacent_edge=ist.num_adjacent_edge, adjacent_edge_abc=ist.adjacent_edge_abc, num_nonz=ist.num_nonz, spmat_data=ist.spmat_data, spmat_indices=ist.spmat_indices, spmat_indptr=ist.spmat_indptr, spmat_ii=ist.spmat_ii, spmat_jj=ist.spmat_jj)
            print("time of caching:", perf_counter()-tic)

    def calc_num_nonz(num_adjacent_edge):
        ist.num_nonz = np.sum(ist.num_adjacent_edge)+num_adjacent_edge.shape[0]
        return ist.num_nonz

    def calc_nnz_each_row(num_adjacent_edge):
        nnz_each_row = ist.num_adjacent_edge[:] + 1
        return nnz_each_row

    def init_A_CSR_pattern(num_adjacent_edge, adjacent_edge):
        num_adj = ist.num_adjacent_edge
        adj = ist.adjacent_edge
        nonz = np.sum(num_adj)+ist.NE
        indptr = np.zeros(ist.NE+1, dtype=np.int32)
        indices = np.zeros(nonz, dtype=np.int32)
        data = np.zeros(nonz, dtype=np.float32)

        indptr[0] = 0
        for i in range(0,ist.NE):
            num_adj_i = num_adj[i]
            indptr[i+1]=indptr[i] + num_adj_i + 1
            indices[indptr[i]:indptr[i+1]-1]= adj[i][:num_adj_i]
            indices[indptr[i+1]-1]=i

        assert indptr[-1] == nonz

        return data, indices, indptr


    def csr_index_to_coo_index(indptr, indices):
        ii, jj = np.zeros_like(indices), np.zeros_like(indices)
        for i in range(ist.NE):
            ii[indptr[i]:indptr[i+1]]=i
            jj[indptr[i]:indptr[i+1]]=indices[indptr[i]:indptr[i+1]]
        return ii, jj


    

    
    def load_cache_initFill_to_cuda(self):
        self.cache_and_initFill()
        extlib.fastFillCloth_set_data(ist.edge.to_numpy(), ist.NE, ist.inv_mass.to_numpy(), ist.NV, ist.pos.to_numpy(), ist.alpha)
        extlib.fastFillCloth_init_from_python_cache(ist.adjacent_edge,
                                            ist.num_adjacent_edge,
                                            ist.adjacent_edge_abc,
                                            ist.num_nonz,
                                            ist.spmat_data,
                                            ist.spmat_indices,
                                            ist.spmat_indptr,
                                            ist.spmat_ii,
                                            ist.spmat_jj,
                                            ist.NE,
                                           ist.NV)




    def compare_find_shared_v_order(v,e1,e2,edge):
        # which is shared v in e1? 0 or 1
        order_in_e1 = 0 if edge[e1][0] == v else 1
        order_in_e2 = 0 if edge[e2][0] == v else 1
        return order_in_e1, order_in_e2


    
    @ti.kernel
    def init_adjacent_edge_abc_kernel(NE:int, edge:ti.template(), adjacent_edge:ti.types.ndarray(), num_adjacent_edge:ti.types.ndarray(), adjacent_edge_abc:ti.types.ndarray()):
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



    def init_adj_edge(edges: np.ndarray):
        # 构建数据结构
        vertex_to_edges = {}
        for edge_index, (v1, v2) in enumerate(edges):
            if v1 not in vertex_to_edges:
                vertex_to_edges[v1] = set()
            if v2 not in vertex_to_edges:
                vertex_to_edges[v2] = set()
            
            vertex_to_edges[v1].add(edge_index)
            vertex_to_edges[v2].add(edge_index)

        # 初始化存储所有边的邻接边的字典
        all_adjacent_edges = {}

        # 查找并存储每条边的邻接边
        for edge_index in range(len(edges)):
            v1, v2 = edges[edge_index]
            adjacent_edges = vertex_to_edges[v1] | vertex_to_edges[v2]  # 合并两个集合
            adjacent_edges.remove(edge_index)  # 移除边本身
            all_adjacent_edges[edge_index] = list(adjacent_edges)

        return all_adjacent_edges, vertex_to_edges

    # # 示例用法
    # edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3]])
    # adjacent_edges_dict = init_adj_edge(edges)
    # print(adjacent_edges_dict)

# for cnt version, require init_A_CSR_pattern() to be called first
@ti.kernel
def fill_A_CSR_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                    indptr:ti.types.ndarray(dtype=ti.i32), 
                    ii:ti.types.ndarray(dtype=ti.i32), 
                    jj:ti.types.ndarray(dtype=ti.i32),
                    adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),
                    num_nonz:ti.i32,
                    alpha:ti.f32):
    for cnt in range(num_nonz):
        i = ii[cnt] # row index
        j = jj[cnt] # col index
        k = cnt - indptr[i] #k-th non-zero element of i-th row. 
        # Because the diag is the final element of each row, 
        # it is also the k-th adjacent edge of i-th edge.
        if i == j: # diag
            data[cnt] = ist.inv_mass[ist.edge[i][0]] + ist.inv_mass[ist.edge[i][1]] + alpha
            continue
        a = adjacent_edge_abc[i, k * 3]
        b = adjacent_edge_abc[i, k * 3 + 1]
        c = adjacent_edge_abc[i, k * 3 + 2]
        g_ab = (ist.pos[a] - ist.pos[b]).normalized()
        g_ac = (ist.pos[a] - ist.pos[c]).normalized()
        offdiag = ist.inv_mass[a] * g_ab.dot(g_ac)
        data[cnt] = offdiag



# legacy
def fill_A_by_spmm(M_inv, ALPHA):
    tic = time.perf_counter()
    G_ii, G_jj, G_vv = np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.float32)
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, ist.gradC, ist.edge)
    G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(ist.NCONS, 3 * ist.NV))
    # print(f"fill_G: {time.perf_counter() - tic:.4f}s")

    tic = time.perf_counter()
    if args.use_withK:
        # Geometric Stiffness: gradG/gradX = M - K, we only use diagonal of K and then replace M_inv with K_inv
        # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
        # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
        # https://blog.csdn.net/weixin_43940314/article/details/139448858
        ist.K_diag.fill(0.0)
        compute_K_kernel(ist.K_diag)
        where_zeros = np.where(M_inv.diagonal()==0)
        mass = 1.0/(M_inv.diagonal()+1e-12)
        MK_inv = scipy.sparse.diags([1.0/(mass - ist.K_diag)], [0], format="dia")
        M_inv = MK_inv # replace old M_inv with MK_inv
        logging.info(f"with K:  max M_inv diag: {np.max(M_inv.diagonal())}, min M_inv diag: {np.min(M_inv.diagonal())}")
        
        M_inv.data[0,where_zeros] = 0.0
        ...

    A = G @ M_inv @ G.transpose() + ALPHA
    A = scipy.sparse.csr_matrix(A)
    print("fill_A_by_spmm  time: ", time.perf_counter() - tic)
    return A, G



def fastFill_fetch():
    extlib.fastFillCloth_fetch_A_data(ist.spmat_data)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(ist.NE, ist.NE))
    return A


# @ti.kernel
# def fill_A_diag_kernel(diags:ti.types.ndarray(dtype=ti.f32), alpha:ti.f32, inv_mass:ti.template(), edge:ti.template()):
#     for i in range(edge.shape[0]):
#         diags[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha


@ti.kernel
def fill_A_ijv_kernel(ii:ti.types.ndarray(dtype=ti.i32), jj:ti.types.ndarray(dtype=ti.i32), vv:ti.types.ndarray(dtype=ti.f32), num_adjacent_edge:ti.types.ndarray(dtype=ti.i32), adjacent_edge:ti.types.ndarray(dtype=ti.i32), adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),  inv_mass:ti.template(), alpha:ti.f32):
    n = 0
    ist.NE = ist.adjacent_edge.shape[0]
    ti.loop_config(serialize=True)
    for i in range(ist.NE): #对每个edge，找到所有的adjacent edge，填充到offdiag，然后填充diag
        for k in range(num_adjacent_edge[i]):
            ia = adjacent_edge[i,k]
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (ist.pos[a] - ist.pos[b]).normalized()
            g_ac = (ist.pos[a] - ist.pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            if offdiag == 0:
                continue
            ii[n] = i
            jj[n] = ia
            vv[n] = offdiag
            n += 1
        # diag
        ii[n] = i
        jj[n] = i
        vv[n] = inv_mass[ist.edge[i][0]] + inv_mass[ist.edge[i][1]] + alpha
        n += 1 



def fill_A_csr_ti(ist):
    fill_A_CSR_kernel(ist.spmat_data, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.adjacent_edge_abc, ist.num_nonz, ist.alpha)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(ist.NE, ist.NE))
    return A

def fill_G():
    tic = time.perf_counter()
    compute_C_and_gradC_kernel(ist.pos, ist.gradC, ist.edge, ist.constraints, ist.rest_len)
    G_ii, G_jj, G_vv = np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.float32)
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, ist.gradC, ist.edge)
    G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(ist.NCONS, 3 * ist.NV))
    print(f"    fill_G: {time.perf_counter() - tic:.4f}s")
    return G

# ---------------------------------------------------------------------------- #
#                                  end fill A                                  #
# ---------------------------------------------------------------------------- #
def fetch_A_from_cuda(lv=0):
    nnz = extlib.fastmg_get_nnz(lv)
    matsize = extlib.fastmg_get_matsize(lv)

    extlib.fastmg_fetch_A(lv, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(matsize, matsize))
    return A

def fetch_A_data_from_cuda():
    extlib.fastmg_fetch_A_data(ist.spmat_data)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(ist.NT, ist.NT))
    return A

def fill_A_in_cuda():
    """Assemble A in cuda end"""
    tic2 = perf_counter()
    if args.use_withK:
        A,G = fill_A_by_spmm(ist.M_inv, ist.ALPHA)
        extlib.fastmg_set_A0(A.data.astype(np.float32), A.indices, A.indptr, A.shape[0], A.shape[1], A.nnz)
    else:
        extlib.fastFillCloth_run(ist.pos.to_numpy())
        extlib.fastmg_set_A0_from_fastFillCloth()
    logging.info(f"    fill_A time: {(perf_counter()-tic2)*1000:.0f}ms")

def get_A0_python()->scipy.sparse.csr_matrix:
    """get A0 from python end for build_P"""
    if args.use_withK:
        A,G = fill_A_by_spmm(ist.M_inv, ist.ALPHA)
    else:
        A = fill_A_csr_ti(ist)
    return A

def get_A0_cuda()->scipy.sparse.csr_matrix:
    """get A0 from cuda end for build_P"""
    if args.use_withK:
        A,G = fill_A_by_spmm(ist.M_inv, ist.ALPHA)
    else:
        fill_A_in_cuda()
        A = fetch_A_from_cuda(0)
    return A

# ---------------------------------------------------------------------------- #
#                                initialization                                #
# ---------------------------------------------------------------------------- #
def init():
    process_dirs(args)
    
    log_level = logging.INFO
    if not args.export_log:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format="%(message)s",filename=args.out_dir + f'/latest.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)


    if args.use_cuda:
        global extlib
        extlib = init_extlib(args,sim="cloth")

    global ist
    ist = Cloth()
    args.frame = ist.frame
    args.ite = ist.ite

    global amg
    if args.solver_type == "AMG":
        if args.use_cuda:
            amg = AmgCuda(
                args=args,
                extlib=extlib,
                get_A0=get_A0_cuda,
                should_setup=should_setup,
                fill_A_in_cuda=fill_A_in_cuda,
                graph_coloring=None,
                copy_A=True,
            )
        else:
            amg = AmgPython(args, get_A0_python, should_setup, copy_A=True)
    if args.solver_type == "AMGX":
        amg = AmgxSolver(args.amgx_config, get_A0_python, args.cuda_dir, args.amgx_lib_dir)
        amg.init()
        ist.amgxsolver = amg
    if args.solver_type == "DIRECT":
        amg = DirectSolver(get_A0_python)

    
    tic_init = time.perf_counter()
    ist.start_date = datetime.datetime.now()
    ist.start_date = ist.start_date.strftime("%Y-%m-%d-%H-%M-%S")
    logging.info(f"start date:{ist.start_date}")

    logging.info("\nInitializing...")
    logging.info("Initializing pos..")
    init_pos(ist.inv_mass,ist.pos, args.N, ist.NV)
    init_tri(ist.tri)
    init_edge(ist.edge, ist.rest_len, ist.pos)
    if args.use_bending:
        ist.tri_pairs, ist.bending_length = init_bending(ist.tri.to_numpy().reshape(-1, 3), ist.pos)
    if args.setup_num == 1:
        init_scale()
    write_mesh(args.out_dir + f"/mesh/{0:04d}", ist.pos.to_numpy(), ist.tri.to_numpy())
    ist.frame = 1
    logging.info("Initializing pos and edge done")

    tic = time.perf_counter()
    if args.solver_type != "XPBD":
        ist.fill_A = FillACloth()
        if args.use_cuda:
            ist.fill_A.load_cache_initFill_to_cuda()
        else:
            ist.fill_A.cache_and_initFill()
    logging.info(f"Init fill time: {time.perf_counter()-tic:.3f}s")

    if args.restart:
        do_restart(args,ist)

    inv_mass_np = np.repeat(ist.inv_mass.to_numpy(), 3, axis=0)
    ist.M_inv = scipy.sparse.diags(inv_mass_np)
    ist.alpha_tilde_np = np.array([ist.alpha] * ist.NCONS)
    ist.ALPHA = scipy.sparse.diags(ist.alpha_tilde_np)

    logging.info(f"Initialization done. Cost time:  {time.perf_counter() - tic_init:.3f}s") 


def export_after_substep():
    ist.tic_export = time.perf_counter()
    if args.export_mesh:
        write_mesh(args.out_dir + f"/mesh/{ist.frame:04d}", ist.pos.to_numpy(), ist.tri.to_numpy())
    if args.export_state:
        save_state(args.out_dir+'/state/' + f"{ist.frame:04d}.npz", ist)
    ist.t_export += time.perf_counter()-ist.tic_export
    ist.t_export_total += ist.t_export
    t_frame = time.perf_counter()-ist.tic_frame
    if args.export_log:
        logging.info(f"Time of exporting: {ist.t_export:.3f}s")
        logging.info(f"Time of frame-{ist.frame}: {t_frame:.3f}s")


def run():
    ist.timer_loop = time.perf_counter()
    ist.initial_frame = ist.frame
    step_pbar = tqdm.tqdm(total=args.end_frame, initial=ist.frame)
    ist.t_export_total = 0.0

    try:
        while True:
            ist.tic_frame = time.perf_counter()
            ist.t_export = 0.0

            if not ist.paused:
                if args.solver_type == "XPBD":
                    substep_xpbd()
                else:
                    substep_all_solver()
                ist.frame += 1
                
                export_after_substep()

            if ist.frame == args.end_frame:
                print("Normallly end.")
                ending(args,ist)
                exit()

            logging.info("\n")
            step_pbar.update(1)
            logging.info("")

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        ending(args,ist)


def main():
    init()
    run()

if __name__ == "__main__":
    main()
