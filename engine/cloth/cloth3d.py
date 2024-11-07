import scipy.sparse
import taichi as ti
import numpy as np
import time
import scipy
import os,sys
import tqdm
import argparse
import logging
import datetime
from time import perf_counter


prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(prj_path)
from engine.file_utils import process_dirs,  do_restart,  export_mat
from engine.init_extlib import init_extlib
from engine.mesh_io import write_mesh
from engine.cloth.bending import init_bending, solve_bending_constraints_xpbd
from engine.solver.amg_python import AmgPython
from engine.solver.amg_cuda import AmgCuda
from engine.solver.amgx_solver import AmgxSolver
from engine.solver.direct_solver import DirectSolver
from engine.solver.iterative_solver import GaussSeidelSolver
from engine.util import ResidualDataAllFrame, ResidualDataOneFrame, ResidualDataOneIter, calc_norm, init_logger



#parse arguments to change default values
from engine.common_args import add_common_args
parser = argparse.ArgumentParser()
parser = add_common_args(parser)

parser.add_argument("-N", type=int, default=64)
parser.add_argument("-compliance", type=float, default=1.0e-8)
parser.add_argument("-setup_num", type=int, default=0, help="attach:0, scale:1")
parser.add_argument("-omega", type=float, default=0.25)
parser.add_argument("-smoother_type", type=str, default="chebyshev")
parser.add_argument("-use_bending", type=int, default=False)


args = parser.parse_args()

if args.setup_num==1: args.gravity = (0.0, 0.0, 0.0)
else : args.gravity = (0.0, -9.8, 0.0)


if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

class Cloth():
    def __init__(self) -> None:
        self.Ps = None
        self.num_levels = 0
        self.paused = False
        self.n_outer_all = [] 
        self.all_stalled = [] 
        self.sim_type = "cloth"
        self.sim_name=f"cloth-N{args.N}"
        self.r_frame = ResidualDataOneFrame([])
        self.r_all = ResidualDataAllFrame([],[])
        self.args = args

        self.frame = 0
        self.ite=0
        
        self.compliance = args.compliance  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
        self.alpha = self.compliance * (1.0 / args.delta_t / args.delta_t)  # timestep related compliance, see XPBD self.paper
        self.alpha_bending = 1.0 * (1.0 / args.delta_t / args.delta_t) #TODO: need to be tuned

        self.build_cloth_mesh()

        if args.use_bending:
            self.tri_pairs, self.bending_length = init_bending(self.tri, self.pos)

        if args.export_strain:
            from engine.cloth.build_cloth_mesh import write_and_rebuild_topology
            self.v2e, self.v2t, self.e2t = write_and_rebuild_topology(self.edge.to_numpy(),self.tri,args.out_dir)

        inv_mass_np = np.repeat(self.inv_mass.to_numpy(), 3, axis=0)
        self.M_inv = scipy.sparse.diags(inv_mass_np)
        self.alpha_tilde_np = np.array([self.alpha] * self.NE)
        self.ALPHA = scipy.sparse.diags(self.alpha_tilde_np)
        

    def build_cloth_mesh(self):
        cloth_type = "quad"
        # cloth_type = "tri"
        # args.cloth_mesh_file = "data/model/tri_cloth/N64.ply"
        from engine.cloth.build_cloth_mesh import TriMeshCloth, QuadMeshCloth
        if cloth_type=="tri":
            mesh = TriMeshCloth(args.cloth_mesh_file)
            mesh.build()
            self.NV, self.NE, self.NT = mesh.NV, mesh.NE, mesh.NT
            self.allocate_fields(self.NV, self.NE, self.NT)
            mesh.fetch_fields(self.pos, self.inv_mass, self.edge, self.rest_len)
            self.tri = mesh.tri
        elif cloth_type=="quad":
            mesh = QuadMeshCloth(args.N, args.setup_num)
            self.NV, self.NE, self.NT = mesh.NV, mesh.NE, mesh.NT
            self.allocate_fields(self.NV, self.NE, self.NT)
            mesh.pass_fields(self.pos, self.inv_mass, self.edge, self.tri_ti, self.rest_len)
            mesh.build()
            self.tri = self.tri_ti.to_numpy().reshape(-1,3)

    
    def allocate_fields(self, NV, NE, NT):
        self.tri_ti      = ti.field(dtype=int, shape=NT*3)
        self.edge        = ti.Vector.field(2, dtype=int, shape=(NE))
        self.pos         = ti.Vector.field(3, dtype=float, shape=(NV))
        self.dpos        = ti.Vector.field(3, dtype=float, shape=(NV))
        self.dpos_withg  = ti.Vector.field(3, dtype=float, shape=(NV))
        self.old_pos     = ti.Vector.field(3, dtype=float, shape=(NV))
        self.vel         = ti.Vector.field(3, dtype=float, shape=(NV))
        self.pos_mid     = ti.Vector.field(3, dtype=float, shape=(NV))
        self.inv_mass    = ti.field(dtype=float, shape=(NV))
        self.rest_len    = ti.field(dtype=float, shape=(NE))
        self.lagrangian  = ti.field(dtype=float, shape=(NE))  
        self.constraints = ti.field(dtype=float, shape=(NE))  
        self.dLambda     = ti.field(dtype=float, shape=(NE))
        self.gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(NE,2)) 
        self.edge_center = ti.Vector.field(3, dtype = ti.float32, shape=(NE))
        self.dual_residual       = ti.field(shape=(NE),    dtype = ti.float32) # -C - alpha * self.lagrangian
        self.predict_pos = ti.Vector.field(3, dtype=float, shape=(NV))
        # self.# primary_residual = np.zeros(dtype=float, shape=(3*NV))
        # self.# K = ti.Matrix.field(3, 3, float, (NV, NV)) 
        # self.# geometric stiffness, only retain diagonal elements
        self.K_diag = np.zeros((NV*3), dtype=float)
        self.current_len = ti.field(ti.f32, shape=NE)
        self.strain = ti.field(ti.f32, shape=NE)
        self.max_strain = 0.0
        self.total_energy = 0.0
        self.potential_energy = 0.0
        self.inertial_energy = 0.0


    def semi_euler(self):
        semi_euler_kernel(self.old_pos, self.inv_mass, self.vel, self.pos, self.predict_pos, args.delta_t)

    def dlam2dpos(self,x):
        transfer_back_to_pos_mfree(x)

    def update_vel(self):
        update_vel_kernel(self.old_pos, self.inv_mass, self.vel, self.pos)

    def compute_C_and_gradC(self):
        compute_C_and_gradC_kernel(self.pos, self.gradC, self.edge, self.constraints, self.rest_len)

    def compute_b(self):
        update_constraints_kernel(self.pos, self.edge, self.rest_len, self.constraints)
        self.b = -self.constraints.to_numpy() - self.alpha_tilde_np * self.lagrangian.to_numpy()
        return self.b
    
    def update_pos(self):
        update_pos_kernel(ist.inv_mass, ist.dpos, ist.pos,args.omega)

    def do_post_iter(self):
        self.r_iter.calc_r(self.frame,self.ite, self.r_iter.tic_iter, self.r_iter.r_Axb)
        export_mat(self, get_A0_cuda, self.b)
        self.r_frame.t_export += self.r_iter.t_export
        logging.info(f"iter time(with export): {(perf_counter()-self.r_iter.tic_iter)*1000:.0f}ms")


    def substep_all_solver(self):
        self.semi_euler()
        self.lagrangian.fill(0)
        self.r_iter.calc_r0()
        for self.ite in range(args.maxiter):
            self.r_iter.tic_iter = perf_counter()
            self.compute_C_and_gradC()
            self.b = self.compute_b()
            x, self.r_iter.r_Axb = linsol.run(self.b)
            self.dlam2dpos(x)
            self.do_post_iter()
            if self.r_iter.check():
                break
        self.n_outer_all.append(self.ite+1)
        self.update_vel()


    def substep_newton(self,newton):
        self.semi_euler()
        self.r_iter.calc_r0()
        for self.ite in range(args.maxiter):
            converge = newton.step_one_iter(self.pos)
            if converge:
                break
        self.n_outer_all.append(self.ite+1)
        self.update_vel()
        self.old_pos.from_numpy(self.pos.to_numpy())


    def substep_xpbd(self):
        self.semi_euler()
        self.lagrangian.fill(0.0)
        self.r_iter.calc_r0()
        for self.ite in range(args.maxiter):
            tic_iter = perf_counter()
            self.dpos.fill(0.0)
            if args.use_bending:
                # TODO: should use seperate dual_residual_bending and lagrangian_bending
                solve_bending_constraints_xpbd(self.dual_residual, self.inv_mass, self.lagrangian, self.dpos, self.pos, self.bending_length, self.tri_pairs, self.alpha_bending)
            solve_distance_constraints_xpbd(self.dual_residual, self.inv_mass, self.edge, self.rest_len, self.lagrangian, self.dpos, self.pos)
            update_pos_kernel(self.inv_mass, self.dpos, self.pos,args.omega)
            self.r_iter.calc_r(self.frame, self.ite, tic_iter)
            if self.r_iter.dual<args.tol:
                break
        self.n_outer_all.append(self.ite+1)
        update_vel_kernel(self.old_pos, self.inv_mass, self.vel, self.pos)
    


@ti.kernel
def semi_euler_kernel(
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
def update_pos_kernel(
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
def update_vel_kernel(
    old_pos:ti.template(),
    inv_mass:ti.template(),    
    vel:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / args.delta_t





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


def calc_primal():
    G = fill_G()
    primary_residual = calc_primary_residual(G, ist.M_inv)
    primal_r = np.linalg.norm(primary_residual).astype(float)
    Newton_r = np.linalg.norm(np.concatenate((ist.dual_residual.to_numpy(), primary_residual))).astype(float)
    return primal_r, Newton_r








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
def calc_strain_cloth_kernel(
    edge:ti.template(),
    rest_len:ti.template(),
    pos:ti.template(),
    NE:ti.i32,
    current_len:ti.template(),
    strain:ti.template(),
):
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        current_len[i] = dis.norm()
        strain[i] = (current_len[i] - rest_len[i])/current_len[i]


def calc_strain()->float:
    calc_strain_cloth_kernel(ist.edge, ist.rest_len, ist.pos, ist.NE, ist.current_len, ist.strain)
    ist.max_strain = np.max(ist.strain.to_numpy())
    return ist.max_strain



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




def transfer_back_to_pos_mfree(x):
    ist.dLambda.from_numpy(x)
    ist.dpos.fill(0.0)
    transfer_back_to_pos_mfree_kernel()
    update_pos_kernel(ist.inv_mass, ist.dpos, ist.pos, args.omega)


def calc_total_energy():
    from engine.util import compute_potential_energy, compute_inertial_energy
    update_constraints_kernel(ist.pos, ist.edge, ist.rest_len, ist.constraints)
    ist.potential_energy = compute_potential_energy(ist)
    ist.inertial_energy = compute_inertial_energy(ist)
    ist.total_energy = ist.potential_energy + ist.inertial_energy
    return ist.total_energy



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


def calc_dual()->float:
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    dual = calc_norm(ist.dual_residual)
    return dual



@ti.kernel
def init_scale():
    scale = 1.5
    for i in range(ist.NV):
        ist.pos[i] *= scale



# ---------------------------------------------------------------------------- #
#                                 start fill A                                 #
# ---------------------------------------------------------------------------- #


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
    extlib.fastFillCloth_fetch_A_data(ist.spmat.data)
    A = scipy.sparse.csr_matrix((ist.spmat.data, ist.spmat.indices, ist.spmat.indptr), shape=(ist.NE, ist.NE))
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
    fill_A_CSR_kernel(ist.spmat.data, ist.spmat.indptr, ist.spmat.ii, ist.spmat.jj, ist.adjacent_edge_abc, ist.num_nonz, ist.alpha)
    A = scipy.sparse.csr_matrix((ist.spmat.data, ist.spmat.indices, ist.spmat.indptr), shape=(ist.NE, ist.NE))
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

    extlib.fastmg_fetch_A(lv, ist.spmat.data, ist.spmat.indices, ist.spmat.indptr)
    A = scipy.sparse.csr_matrix((ist.spmat.data, ist.spmat.indices, ist.spmat.indptr), shape=(matsize, matsize))
    return A

def fetch_A_data_from_cuda():
    extlib.fastmg_fetch_A_data(ist.spmat.data)
    A = scipy.sparse.csr_matrix((ist.spmat.data, ist.spmat.indices, ist.spmat.indptr), shape=(ist.NT, ist.NT))
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
def init_linear_solver(args):
    if args.solver_type == "AMG":
        if args.use_cuda:
            linsol = AmgCuda(
                args=args,
                extlib=extlib,
                get_A0=get_A0_cuda,
                should_setup=should_setup,
                fill_A_in_cuda=fill_A_in_cuda,
                graph_coloring=None,
                copy_A=True,
            )
        else:
            linsol = AmgPython(args, get_A0_python, should_setup, copy_A=True)
    elif args.solver_type == "AMGX":
        linsol = AmgxSolver(args.amgx_config, get_A0_python, args.cuda_dir, args.amgx_lib_dir)
        linsol.init()
        ist.amgxsolver = linsol
    elif args.solver_type == "DIRECT":
        linsol = DirectSolver(get_A0_python)
    elif args.solver_type == "GS":
        linsol = GaussSeidelSolver(get_A0_python, args)
    elif args.solver_type == "XPBD":
        linsol = None
    else:
        linsol = None
    return linsol


def init_r_iter(args, ist):
    ist.r_iter = ResidualDataOneIter(args, 
                                    calc_dual=calc_dual,
                                    calc_primal=calc_primal, calc_total_energy=calc_total_energy, calc_strain=calc_strain)


def init_fill():
    tic = time.perf_counter()
    if args.solver_type != "XPBD":
        from engine.cloth.fill_A import FillACloth
        fill_A = FillACloth(ist.pos, ist.inv_mass, ist.edge, ist.alpha,  args.use_cache, args.use_cuda, extlib)
        fill_A.init()
        ist.spmat = fill_A.spmat
        ist.adjacent_edge_abc = fill_A.adjacent_edge_abc
        ist.num_adjacent_edge = fill_A.num_adjacent_edge
        ist.num_nonz = fill_A.num_nonz
    logging.info(f"Init fill time: {time.perf_counter()-tic:.3f}s")


def init():
    tic_init = time.perf_counter()

    process_dirs(args)
    
    init_logger(args)

    global extlib
    extlib = init_extlib(args,sim="cloth")

    global ist
    ist = Cloth()
    ist.delta_t = args.delta_t

    ist.start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.info(f"start date:{ist.start_date}")

    global linsol
    linsol = init_linear_solver(args)

    if args.solver_type == "NEWTON":
        from engine.cloth.newton_method import NewtonMethod
        ist.newton = NewtonMethod(ist)

    init_r_iter(args, ist)

    if args.setup_num == 1:
        init_scale()
    write_mesh(args.out_dir + f"/mesh/{0:04d}", ist.pos.to_numpy(), ist.tri)

    init_fill()

    ist.frame = 1
    if args.restart:
        do_restart(args,ist) #will change frame number
    logging.info(f"Initialization done. Cost time:  {time.perf_counter() - tic_init:.3f}s") 



def main():
    init()
    from engine.util import main_loop
    main_loop(ist, args)

if __name__ == "__main__":
    main()
