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
from pathlib import Path
import taichi.math as tm
from scipy.io import mmwrite, mmread


prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(prj_path)
from engine.file_utils import process_dirs,  do_restart
from engine.init_extlib import init_extlib
from engine.mesh_io import write_mesh
from engine.cloth.bending import init_bending, solve_bending_constraints_xpbd, add_distance_constraints_from_tri_pairs
from engine.solver.amg_python import AmgPython
from engine.solver.amg_cuda import AmgCuda
from engine.solver.amgx_solver import AmgxSolver
from engine.solver.direct_solver import DirectSolver
from engine.solver.iterative_solver import GaussSeidelSolver
from engine.util import norm_sqr, calc_norm, init_logger, do_post_iter, timeit, set_gravity_as_force, ResidualDataOneIter
from engine.physical_base import PhysicalBase
from engine.line_search import LineSearch
from engine.physical_data import PhysicalData



def init_args():
    #parse arguments to change default values
    from engine.common_args import add_common_args
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)

    parser.add_argument("-N", type=int, default=64)
    parser.add_argument("-compliance", type=float, default=1.0e-8)
    parser.add_argument("-compliance_bending", type=float, default=1.0e-8)
    parser.add_argument("-setup_num", type=int, default=0, help="attach:0, scale:1")
    parser.add_argument("-omega", type=float, default=0.25)
    parser.add_argument("-smoother_type", type=str, default="chebyshev")
    parser.add_argument("-use_bending", type=int, default=False)
    parser.add_argument("-cloth_mesh_file", type=str, default="./data/model/tri_cloth/tri_cloth.obj")
    parser.add_argument("-cloth_mesh_type", type=str, default="quad", choices=["quad", "tri", "txt"])
    # ./data/model/tri_cloth/tri_cloth.obj
    # ./data/model/tri_cloth/N64.ply
    parser.add_argument("-pos_file", type=str, default="data/model/fast_mass_spring/pos.txt")
    parser.add_argument("-edge_file", type=str, default="data/model/fast_mass_spring/edge.txt")
    parser.add_argument("-tri_file", type=str, default="data/model/fast_mass_spring/tri.txt")
    parser.add_argument("-use_initFill", type=int, default=False)
    parser.add_argument("-write_physdata", type=int, default=False)

    args = parser.parse_args()

    if args.setup_num==1: args.gravity = (0.0, 0.0, 0.0)
    else : args.gravity = (0.0, -9.8, 0.0)
    return args



@ti.data_oriented
class Cloth(PhysicalBase):
    def __init__(self,args, extlib) -> None:
        super().__init__()
        self.args = args
        self.extlib = extlib
        self.sim_type = "cloth"

        # ---------------------------------------------------------------------------- #
        #                               mesh and topology                              #
        # ---------------------------------------------------------------------------- #
        self.build_mesh()
        self.init_constraints(self.edge, self.pos,self.tri)
        self.init_dynamics(self.NV)
        self.init_physics(args.N, args.setup_num, self.NV)

        if args.export_strain:
            self.max_strain = 0.0
            self.strain = ti.field(ti.f32, shape=self.NCONS)
            from engine.cloth.build_cloth_mesh import write_and_rebuild_topology
            self.v2e, self.v2t, self.e2t = write_and_rebuild_topology(self.edge.to_numpy(),self.tri,args.out_dir)

        self.linsol = self.init_linear_solver(args, extlib)

        if args.setup_num == 1:
            from engine.ti_kernels import init_scale
            init_scale(self.NV, self.pos, 1.5)

        if args.solver_type == "AMG":
            self.args.use_initFill = True
        if self.args.use_initFill:
            self.init_fill()


    def set_pin(self, NV, N, setup_num):
        pin = np.zeros(NV, dtype=np.int32)
        if setup_num == 0:
            pin_idx = [N, NV-1]
            pin[pin_idx] = 1

        self.pin = pin
        return pin

    def set_mass(self, NV, pin):
        mass = np.zeros(NV, dtype=np.float32)
        inv_mass_np = np.zeros(NV, dtype=np.float32)
        mass[:]=1.0
        mass[pin!=0] = 0.0 
        inv_mass_np[:] = 1.0
        inv_mass_np[pin!=0] = 0.0 
        
        inv_mass3 = np.repeat(inv_mass_np, 3, axis=0)
        M_inv = scipy.sparse.diags(inv_mass3)
        mass3 = np.repeat(mass, 3, axis=0)
        MASS = scipy.sparse.diags(mass3, format="csr")
        self.MASS = MASS
        self.M_inv = M_inv
        self.inv_mass    = ti.field(dtype=float, shape=(NV))
        self.inv_mass.from_numpy(inv_mass_np)


    def init_physics(self, N, setup_num, NV):
        pin = self.set_pin(NV,N,setup_num)
        self.set_mass(NV,pin)

        
    def build_mesh(self,):
        # cloth_type = "quad" or
        # cloth_type = "tri"
        # args.cloth_mesh_file = "data/model/tri_cloth/N64.ply"
        from engine.cloth.build_cloth_mesh import TriMeshCloth, QuadMeshCloth, TriMeshClothTxt
        if args.cloth_mesh_type=="tri":
            mesh = TriMeshCloth(args.cloth_mesh_file)
            name = Path(args.cloth_mesh_file).name
            self.sim_name=f"cloth-{name}"
        elif args.cloth_mesh_type=="quad":
            mesh = QuadMeshCloth(args.N)
            self.sim_name=f"cloth-N{args.N}"
        if args.cloth_mesh_type=="txt":
            mesh = TriMeshClothTxt(args.pos_file, args.edge_file, args.tri_file)
            self.sim_name=f"cloth-txt"
        pos, edge, tri = mesh.build()

        self.NV, self.NE, self.NT = mesh.NV, mesh.NE, mesh.NT
        self.pos = ti.field(dtype=tm.vec3, shape=self.NV)
        self.edge = ti.field(dtype=tm.ivec2, shape=self.NE)
        self.pos.from_numpy(pos)
        self.edge.from_numpy(edge)
        self.tri = tri
    

    def init_dynamics(self, NV):
        self.dpos        = ti.Vector.field(3, dtype=float, shape=(NV))
        self.old_pos     = ti.Vector.field(3, dtype=float, shape=(NV))
        self.vel         = ti.Vector.field(3, dtype=float, shape=(NV))
        self.pos_mid     = ti.Vector.field(3, dtype=float, shape=(NV))
        self.inv_mass    = ti.field(dtype=float, shape=(NV))
        self.predict_pos = ti.Vector.field(3, dtype=float, shape=(NV))

        self.energy = 0.0
        self.potential_energy = 0.0
        self.inertial_energy = 0.0
        self.K_diag = np.zeros((NV*3), dtype=float)


    def init_constraints(self,vert,pos,tri):
        from engine.cloth.constraints import ClothConstraints
        self.cons = ClothConstraints(self.args,vert,pos,tri)

        self.dc = self.cons.disConsAos

        self.gradC = self.cons.gradC

        self.rest_len = self.dc.rest_len
        self.alpha_tilde = self.dc.alpha_tilde
        self.lagrangian = self.dc.lam
        self.dLambda = self.dc.dlam
        self.dual_residual = self.dc.dualr
        self.constraints = self.dc.c



    def semi_euler(self):
        semi_euler_kernel(self.old_pos, self.inv_mass, self.vel, self.pos, self.predict_pos, args.delta_t)


                
    def dlam2dpos(self,dlam):
        self.dLambda.from_numpy(dlam)
        dlam2dpos_kernel(self.edge, self.inv_mass, self.dLambda, self.lagrangian, self.gradC, self.dpos)

    def update_vel(self):
        update_vel_kernel(self.old_pos, self.inv_mass, self.vel, self.pos)

    def compute_C_and_gradC(self):
        compute_C_and_gradC_kernel(self.pos, self.gradC, self.edge, self.constraints, self.rest_len)

    def compute_b(self):
        self.update_constraints()
        self.b = -self.constraints.to_numpy() - self.alpha_tilde.to_numpy() * self.lagrangian.to_numpy()
        return self.b
    
    def update_pos(self):
        update_pos_kernel(self.inv_mass, self.dpos, self.pos,args.omega)
        
    def update_constraints(self):
        update_constraints_kernel(self.pos, self.edge, self.rest_len, self.constraints)


    def calc_dual(self)->float:
        calc_dual_residual(self.dual_residual, self.edge, self.rest_len, self.lagrangian, self.pos, self.alpha_tilde)
        dual = calc_norm(self.dual_residual)
        return dual
    
    def calc_strain(self)->float:
        calc_strain_cloth_kernel(self.edge, self.rest_len, self.pos, self.strain)
        self.max_strain = np.max(self.strain.to_numpy())
        return self.max_strain


    def fill_G(self):
        tic = time.perf_counter()
        compute_C_and_gradC_kernel(self.pos, self.gradC, self.edge, self.constraints, self.rest_len)
        G_ii, G_jj, G_vv = np.zeros(self.NCONS*6, dtype=np.int32), np.zeros(self.NCONS*6, dtype=np.int32), np.zeros(self.NCONS*6, dtype=np.float32)
        fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, self.gradC, self.edge)
        G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(self.NCONS, 3 * self.NV))
        print(f"    fill_G: {time.perf_counter() - tic:.4f}s")
        self.G = G
        return G

    def substep_all_solver(self):
        self.semi_euler()
        self.lagrangian.fill(0)
        self.r_iter.calc_r0()
        for self.ite in range(args.maxiter):
            self.r_iter.tic_iter = perf_counter()
            self.compute_C_and_gradC()
            self.b = self.compute_b()
            dlambda, self.r_iter.r_Axb = self.linsol.run(self.b)
            self.dlam2dpos(dlambda)
            self.update_pos()
            do_post_iter(self, self.get_A0_cuda)
            if self.r_iter.check():
                break
        self.n_outer_all.append(self.ite+1)
        self.update_vel()


    def project_constraints_xpbd(self):
        if args.use_bending:
            # TODO: should use seperate dual_residual_bending and lagrangian_bending
            solve_bending_constraints_xpbd(self.dual_residual, self.inv_mass, self.lagrangian, self.dpos, self.pos, self.bending_length, self.tri_pairs, self.alpha_bending)
        solve_distance_constraints_xpbd(self.dual_residual, self.inv_mass, self.edge, self.rest_len, self.lagrangian, self.dpos, self.pos, self.alpha_tilde)

    def substep_xpbd(self):
        self.semi_euler()
        self.lagrangian.fill(0)
        self.do_pre_iter0()
        for self.ite in range(args.maxiter):
            self.r_iter.tic_iter = perf_counter()
            self.project_constraints_xpbd()
            self.do_post_iter_xpbd()
            if self.r_iter.check():
                break
        self.collision_response()
        self.n_outer_all.append(self.ite+1)
        self.update_vel()


    def step_one_iter_mgpbd(self):
        """
        One iter of mgpbd
        """
        self.compute_C_and_gradC()
        self.b = self.compute_b()
        A = self.assemble_A()
        dlambda, self.r_iter.r_Axb = self.linsol.run(self.b)
        self.dlam2dpos(dlambda)
        self.update_pos()


    def assemble_A(self):
        # taichi csr version 
        A = self.fill_A_csr_ti()
        # spmm version
        # A = self.fill_A_by_spmm(self.M_inv, self.ALPHA_TILDE)
        # cuda fetch version
        # A = self.get_A0_cuda()
        # cuda no-fetch version
        # self.fill_A_in_cuda()


    def fastFill_fetch(self):
        self.extlib.fastFillCloth_fetch_A_data(self.spmat.data)
        A = scipy.sparse.csr_matrix((self.spmat.data, self.spmat.indices, self.spmat.indptr), shape=(self.NCONS, self.NCONS))
        return A

    def fetch_A_from_cuda(self,lv=0):
        nnz = self.extlib.fastmg_get_nnz(lv)
        matsize = self.extlib.fastmg_get_matsize(lv)

        self.extlib.fastmg_fetch_A(lv, self.spmat.data, self.spmat.indices, self.spmat.indptr)
        A = scipy.sparse.csr_matrix((self.spmat.data, self.spmat.indices, self.spmat.indptr), shape=(matsize, matsize))
        return A

    def fetch_A_data_from_cuda(self):
        self.extlib.fastmg_fetch_A_data(self.spmat.data)
        A = scipy.sparse.csr_matrix((self.spmat.data, self.spmat.indices, self.spmat.indptr), shape=(self.NT, self.NT))
        return A

    def fill_A_in_cuda(self):
        """Assemble A in cuda end"""
        tic2 = perf_counter()
        if args.use_withK:
            A,G = self.fill_A_by_spmm(self.M_inv, self.ALPHA_TILDE)
            self.extlib.fastmg_set_A0(A.data.astype(np.float32), A.indices, A.indptr, A.shape[0], A.shape[1], A.nnz)
        else:
            self.extlib.fastFillCloth_run(self.pos.to_numpy())
            self.extlib.fastmg_set_A0_from_fastFillCloth()
        logging.info(f"    fill_A time: {(perf_counter()-tic2)*1000:.0f}ms")

    def get_A0_python(self)->scipy.sparse.csr_matrix:
        """get A0 from python end for build_P"""
        if args.use_withK:
            A,G = self.fill_A_by_spmm(self.M_inv, self.ALPHA_TILDE)
        else:
            A = self.fill_A_csr_ti()
        return A

    def get_A0_cuda(self)->scipy.sparse.csr_matrix:
        """get A0 from cuda end for build_P"""
        if args.use_withK:
            A,G = self.fill_A_by_spmm(self.M_inv, self.ALPHA_TILDE)
        else:
            self.fill_A_in_cuda()
            A = self.fetch_A_from_cuda(0)
        return A

    def init_linear_solver(self, args, extlib):
        if args.linsol_type == "AMG":
            if args.use_cuda:
                linsol = AmgCuda(
                    args=args,
                    extlib=extlib,
                    get_A0=self.get_A0_cuda,
                    should_setup=self.should_setup,
                    fill_A_in_cuda=self.fill_A_in_cuda,
                    graph_coloring=None,
                    copy_A=True,
                )
            else:
                linsol = AmgPython(args, self.get_A0_python, self.should_setup, copy_A=True)
        elif args.linsol_type == "AMGX":
            linsol = AmgxSolver(args.amgx_config, self.get_A0_python, args.cuda_dir, args.amgx_lib_dir)
        elif args.linsol_type == "DIRECT":
            linsol = DirectSolver(self.get_A0_python)
        elif args.linsol_type == "GS":
            linsol = GaussSeidelSolver(self.get_A0_python, args)
        else:
            linsol = None
        return linsol


    def init_fill(self):
        if args.solver_type == "XPBD" :
            return
        tic = time.perf_counter()
        from engine.cloth.fill_A import FillACloth
        alpha_tilde_constant = args.compliance/args.delta_t/args.delta_t
        fill_A = FillACloth(self.pos, self.inv_mass, self.edge,
         alpha_tilde_constant,  args.use_cache, args.use_cuda, self.extlib)
        fill_A.init()
        self.spmat = fill_A.spmat
        self.adjacent_edge_abc = fill_A.adjacent_edge_abc
        self.num_adjacent_edge = fill_A.num_adjacent_edge
        self.num_nonz = fill_A.num_nonz
        logging.info(f"Init fill time: {time.perf_counter()-tic:.3f}s")


    def fill_A_csr_ti(self):
        self.fill_A_CSR_kernel(self.spmat.data, self.spmat.indptr,
                               self.spmat.ii, self.spmat.jj,
                               self.adjacent_edge_abc,
                               self.num_nonz, self.alpha_tilde,
                               self.pos, self.edge, self.inv_mass)
                               
        A = scipy.sparse.csr_matrix((self.spmat.data, self.spmat.indices,
                                     self.spmat.indptr),
                                     shape=(self.NCONS, self.NCONS))
        return A
    

    # for cnt version, require init_A_CSR_pattern() to be called first
    @staticmethod
    @ti.kernel
    def fill_A_CSR_kernel(self,
                        data:ti.types.ndarray(dtype=ti.f32), 
                        indptr:ti.types.ndarray(dtype=ti.i32), 
                        ii:ti.types.ndarray(dtype=ti.i32), 
                        jj:ti.types.ndarray(dtype=ti.i32),
                        adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),
                        num_nonz:ti.i32,
                        alpha_tilde:ti.template(),
                        pos:ti.template(),
                        edge:ti.template(),
                        inv_mass:ti.template()
                        ):
        for cnt in range(num_nonz):
            i = ii[cnt] # row index
            j = jj[cnt] # col index
            k = cnt - indptr[i] #k-th non-zero element of i-th row. 
            # Because the diag is the final element of each row, 
            # it is also the k-th adjacent edge of i-th edge.
            if i == j: # diag
                data[cnt] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha_tilde[i]
                continue
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            data[cnt] = offdiag


    def fill_G(self):
        tic = time.perf_counter()
        compute_C_and_gradC_kernel(self.pos, self.gradC, self.edge, self.constraints, self.rest_len)
        G_ii, G_jj, G_vv = np.zeros(self.NCONS*6, dtype=np.int32), np.zeros(self.NCONS*6, dtype=np.int32), np.zeros(self.NCONS*6, dtype=np.float32)
        fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, self.gradC, self.edge)
        G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(self.NCONS, 3 * self.NV))
        print(f"    fill_G: {time.perf_counter() - tic:.4f}s")
        return G

    # legacy
    def fill_A_by_spmm(self, M_inv, ALPHA_TILDE):
        tic = time.perf_counter()
        G_ii, G_jj, G_vv = np.zeros(self.NCONS*6, dtype=np.int32), np.zeros(self.NCONS*6, dtype=np.int32), np.zeros(self.NCONS*6, dtype=np.float32)
        fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, self.gradC, self.edge)
        G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(self.NCONS, 3 * self.NV))
        # print(f"fill_G: {time.perf_counter() - tic:.4f}s")

        tic = time.perf_counter()
        if args.use_withK:
            # Geometric Stiffness: gradG/gradX = M - K, we only use diagonal of K and then replace M_inv with K_inv
            # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
            # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
            # https://blog.csdn.net/weixin_43940314/article/details/139448858
            self.K_diag.fill(0.0)
            compute_K_kernel(self.K_diag)
            where_zeros = np.where(M_inv.diagonal()==0)
            mass = 1.0/(M_inv.diagonal()+1e-12)
            MK_inv = scipy.sparse.diags([1.0/(mass - self.K_diag)], [0], format="dia")
            M_inv = MK_inv # replace old M_inv with MK_inv
            logging.info(f"with K:  max M_inv diag: {np.max(M_inv.diagonal())}, min M_inv diag: {np.min(M_inv.diagonal())}")
            
            M_inv.data[0,where_zeros] = 0.0
            ...

        A = G @ M_inv @ G.transpose() + ALPHA_TILDE
        A = scipy.sparse.csr_matrix(A)
        print("fill_A_by_spmm  time: ", time.perf_counter() - tic)
        return A, G


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
    for i in range(pos.shape[0]):
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
    alpha_tilde:ti.template(),
):
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian[i] * alpha_tilde[i]) / (invM0 + invM1 + alpha_tilde[i])
        lagrangian[i] += delta_lagrangian

        # residual
        dual_residual[i] = -(constraint + alpha_tilde[i] * lagrangian[i])
        
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
    for i in range(inv_mass.shape[0]):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]




@ti.kernel
def update_vel_kernel(
    old_pos:ti.template(),
    inv_mass:ti.template(),    
    vel:ti.template(),
    pos:ti.template(),
):
    for i in range(pos.shape[0]):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / args.delta_t





@ti.kernel
def calc_dual_residual(
    dual_residual: ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    pos:ti.template(),
    alpha_tilde:ti.template(),
):
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]

        # residual(lagrangian=0 for first iteration)
        dual_residual[i] = -(constraint + alpha_tilde[i] * lagrangian[i])


def calc_primary_residual(G,M_inv):
    MASS = scipy.sparse.diags(1.0/(M_inv.diagonal()), format="csr")
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


# # FIXME: Why it is 10x higher than v1
# @ti.kernel
# def calc_primal_v2_kernel(
#     edge:ti.template(),
#     pos:ti.template(),
#     predict_pos:ti.template(),
#     lagrangian:ti.template(),
#     inv_mass:ti.template(),
# )->ti.f32:
#     """
#     res = M(x-x~) - G^T * lambda
#     The first  part is 3nx1
#     The second part is (3nxm) x (mx1) = 3nx1
#     For each edge, we have two points, so each i is 6x1 dimension
#     [x0]   [g0x]
#     [y0]   [g0y]
#     [z0] + [g0z]
#     [x1]   [g1x]
#     [y1]   [g1y]
#     [z1]   [g1z]
#     first add vec, then take norm for all edges
#     Or individually calc norm sqr of each points finally sqrt.
#     res <= norm(res)
#     """
#     res = 0.0
#     for i in range(edge.shape[0]):
#         idx0, idx1 = edge[i]
#         dis = pos[idx0] - pos[idx1]
#         g = dis.normalized()

#         # vertex 0 
#         vec1 = lagrangian[i] * g    #small 3x1 vector
#         if inv_mass[idx0] != 0.0:
#             vec1 += 1.0/inv_mass[idx0] * (pos[idx0] - predict_pos[idx0])
#         q1 = vec1.norm_sqr()

#         # vertex 1
#         vec2 = lagrangian[i] * (-g)
#         if inv_mass[idx1] != 0.0:
#             vec2 += 1.0/inv_mass[idx1] * (pos[idx1] - predict_pos[idx1])
#         q2 = vec2.norm_sqr()

#         res += (q1 + q2)
#     res = ti.sqrt(res)
#     return res

# # FIXME: Why it is 10x higher than v1
# def calc_primal_v2():
#     dual = np.linalg.norm(ist.dual_residual.to_numpy())
#     primal = calc_primal_v2_kernel(ist.edge,ist.pos,ist.predict_pos,ist.lagrangian,ist.inv_mass)
#     Newton_r = np.sqrt(dual**2 + primal**2)
#     return primal, Newton_r





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
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]
        g = dis.normalized()

        gradC[i, 0] = g
        gradC[i, 1] = -g


@ti.kernel
def compute_K_kernel(K_diag:ti.types.ndarray(),):
    for i in range(ist.edge.shape[0]):
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
    vert:ti.template(),
    rest_len:ti.template(),
    constraints:ti.template(),
):
    for i in range(rest_len.shape[0]):
        idx0, idx1 = vert[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]


@ti.kernel
def calc_strain_cloth_kernel(
    edge:ti.template(),
    rest_len:ti.template(),
    pos:ti.template(),
    strain:ti.template(),
):
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        l = dis.norm()
        strain[i] = (l - rest_len[i])/rest_len[i]





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
def dlam2dpos_kernel(
    edge:ti.template(),
    inv_mass:ti.template(),
    dLambda:ti.template(),
    lagrangian:ti.template(),
    gradC:ti.template(),
    dpos:ti.template(),
):
    for i in range(dpos.shape[0]):
        dpos[i] = ti.Vector([0.0, 0.0, 0.0])
    
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        lagrangian[i] += dLambda[i]
        gradient = gradC[i, 0]
        if invM0 != 0.0:
            dpos[idx0] += invM0 * dLambda[i] * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * dLambda[i] * gradient



# ---------------------------------------------------------------------------- #
#                                 start fill A                                 #
# ---------------------------------------------------------------------------- #

@ti.kernel
def fill_A_diag_kernel(diags:ti.types.ndarray(dtype=ti.f32), alpha_tilde:ti.f32, inv_mass:ti.template(), edge:ti.template()):
    for i in range(edge.shape[0]):
        diags[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha_tilde


@ti.kernel
def fill_A_ijv_kernel(ii:ti.types.ndarray(dtype=ti.i32),
                    jj:ti.types.ndarray(dtype=ti.i32),
                    vv:ti.types.ndarray(dtype=ti.f32),
                    num_adjacent_edge:ti.types.ndarray(dtype=ti.i32),
                    adjacent_edge:ti.types.ndarray(dtype=ti.i32),
                    adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),
                    inv_mass:ti.template(),
                    alpha_tilde:ti.template()):
    n = 0
    ti.loop_config(serialize=True)
    for i in range(adjacent_edge.shape[0]): #对每个edge，找到所有的adjacent edge，填充到offdiag，然后填充diag
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
        vv[n] = inv_mass[ist.edge[i][0]] + inv_mass[ist.edge[i][1]] + alpha_tilde[i]
        n += 1 


# ---------------------------------------------------------------------------- #
#                                  end fill A                                  #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                initialization                                #
# ---------------------------------------------------------------------------- #
def init():
    tic_init = time.perf_counter()
    global args
    args = init_args()

    if args.arch == "gpu":
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)

    process_dirs(args)
    init_logger(args)

    global extlib
    extlib = init_extlib(args,sim="cloth")

    global ist
    if args.solver_type == "NEWTON":
        from engine.cloth.newton_method import TestNewtonMethod
        ist = TestNewtonMethod(args, extlib)
    else:
        ist = Cloth(args, extlib)

    if args.restart:
        do_restart(args,ist) #will change frame number
    else:
        pos_np = ist.pos.to_numpy() if type(ist.pos) != np.ndarray else ist.pos
        write_mesh(args.out_dir + f"/mesh/{0:04d}", pos_np, ist.tri)

    ist.r_iter = ResidualDataOneIter(
            calc_dual=ist.calc_dual,
            calc_primal =ist.calc_primal,
            calc_energy =ist.calc_energy,
            calc_strain =ist.calc_strain,
            tol=args.tol,
            rtol=args.rtol,
            converge_condition=args.converge_condition,
            args = args)
        
    logging.info(f"Initialization done. Cost time:  {time.perf_counter() - tic_init:.3f}s") 



def main():
    init()
    from engine.util import main_loop
    main_loop(ist, args)

if __name__ == "__main__":
    main()
