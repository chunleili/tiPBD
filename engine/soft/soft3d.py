import taichi as ti
from taichi.lang.ops import sqrt
import numpy as np
import logging
from logging import info
import scipy
import sys, os, argparse
import time
from time import perf_counter
from pathlib import Path
from collections import namedtuple
import json
from functools import singledispatch
import ctypes
import numpy.ctypeslib as ctl
import datetime

prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(prj_path)
from engine.file_utils import process_dirs
from engine.mesh_io import write_mesh, read_tet
from engine.common_args import add_common_args
from engine.init_extlib import init_extlib
from engine.solver.amg_python import AmgPython
from engine.solver.amg_cuda import AmgCuda
from engine.solver.amgx_solver import AmgxSolver
from engine.solver.direct_solver import DirectSolver
from engine.util import calc_norm,  ResidualDataOneIter, do_post_iter, init_logger, timeit
from engine.physical_base import PhysicalBase
from script.convert.geo import Geo

parser = argparse.ArgumentParser()

parser = add_common_args(parser)

parser.add_argument("-mu", type=float, default=1e6)
parser.add_argument("-damping_coeff", type=float, default=1.0)
parser.add_argument("-total_mass", type=float, default=16000.0)
parser.add_argument("-model_path", type=str, default=f"data/model/ball/ball.node")
# "data/model/cube/minicube.node"
# "data/model/bunny1k2k/coarse.node"
# "data/model/bunny_small/bunny_small.node"
# "data/model/bunnyBig/bunnyBig.node"
# "data/model/bunny85w/bunny85w.node"
# "data/model/ball/ball.node"
parser.add_argument("-reinit", type=str, default="freefall")
parser.add_argument("-large", action="store_true")
parser.add_argument("-small", action="store_true")
parser.add_argument("-use_ground_collision", type=int, default=0)
parser.add_argument("-omega", type=float, default=0.1)
parser.add_argument("-smoother_type", type=str, default="jacobi")


args = parser.parse_args()

if args.large:
    args.model_path = f"data/model/bunny85w/bunny85w.node"
if args.small:
    args.model_path = f"data/model/bunny1k2k/coarse.node"

if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

@ti.data_oriented
class SoftBody(PhysicalBase):
    def __init__(self, mesh_file):
        super().__init__()

        self.r_iter = ResidualDataOneIter(
                        calc_dual   = self.calc_dual,
                        calc_primal = self.calc_primal,
                        calc_energy = self.calc_energy,
                        calc_strain = self.calc_strain,
                        tol=args.tol,
                        rtol=args.rtol,
                        converge_condition=args.converge_condition,
                        args = args,
                                    )

        self.args = args
        self.delta_t = args.delta_t
        self.omega = args.omega
        self.gravity = ti.Vector(args.gravity)
        self.damping_coeff = args.damping_coeff


        dir = str(Path(mesh_file).parent.stem)
        self.sim_name = f"soft3d-{dir}-{str(Path(mesh_file).stem)}"


        if args.use_pintoanimation:
            self.read_geo()
            self.reinit()
        else:
            self.build_mesh(mesh_file)
            self.NCONS = self.NT
            self.allocate_fields(self.NV, self.NT)
            self.initialize()
            self.reinit()
            if args.export_mesh:
                write_mesh(args.out_dir + f"/mesh/{self.frame:04d}", self.pos.to_numpy(), self.model_tri)

        self.force = np.zeros((self.NV, 3), dtype=np.float32)
        self.state = [self.pos,]

        if args.solver_type != "XPBD" and args.solver_type != "NEWTON":
            from engine.soft.fill_A import init_direct_fill_A
            init_direct_fill_A(self,extlib)

        self.linsol = init_linear_solver()

        info(f"Creating instance done")


    def read_geo_pinpos(self):
        dir = prj_path + "/" + "data/model/pintoanimation/"
        geo = Geo(dir+f"physdata_{ist.frame}.geo")
        pinpos = np.array(geo.get_pos())
        assert pinpos.shape[0] == self.pos.shape[0]
        # set_pinpos_kernel(self.pin, self.pos, pinpos)
        pos_ = self.pos.to_numpy()
        pos_[self.pin] = pinpos[self.pin]
        self.pos.from_numpy(pos_)

    def read_geo(self, input=None):
        dir = prj_path + "/" + "data/model/pintoanimation/"
        geo = Geo(dir+"physdata_78.geo") #TODO:should be every frame
        pin = np.array(geo.get_gluetoaniamtion(),dtype=np.bool_)
        vert = np.array(geo.get_vert(),dtype=np.int32)
        pinpos = np.array(geo.get_pos())

        self.NV = pinpos.shape[0]
        self.NT = vert.shape[0]
        self.NCONS = self.NT
        self.allocate_fields(self.NV, self.NT)

        self.pin = pin
        self.vert = vert
        self.pinpos = pinpos
        self.pos.from_numpy(pinpos)
        self.pos_mid.from_numpy(pinpos)
        self.old_pos.from_numpy(pinpos)
        self.tet_indices.from_numpy(vert)
        self.geodir = dir
        self.geo = geo
        
        # init physics
        inv_mass_np = np.ones(self.NV, dtype=np.float32)
        inv_mass_np[pin] = 0.0
        self.inv_mass.from_numpy(inv_mass_np)
        init_B(self.pos, self.tet_indices, self.B)
        init_alpha_tilde(self.pos, self.tet_indices, self.rest_volume, self.alpha_tilde, 1.0 / args.mu, 1.0 / args.delta_t / args.delta_t)
        self.alpha_tilde_np = self.alpha_tilde.to_numpy()

    def write_geo(self, output=None):
        self.geo.set_positions(self.pos.to_numpy())
        self.geo.write(self.geodir+f"physdata_{self.frame}_out.geo")

    def build_mesh(self,mesh_file):
        tic = time.perf_counter()
        self.model_pos, self.model_tet, self.model_tri = read_tet(mesh_file, build_face_flag=True)
        print(f"read_tet cost: {time.perf_counter() - tic:.4f}s")
        self.NV = len(self.model_pos)
        self.NT = len(self.model_tet)
        self.NF = len(self.model_tri)
        self.display_indices = ti.field(ti.i32, self.NF * 3)
        self.display_indices.from_numpy(self.model_tri.flatten())
        self.tri = self.model_tri.copy()

    def allocate_fields(self, NV, NT):
        self.pos = ti.Vector.field(3, float, NV)
        self.pos_mid = ti.Vector.field(3, float, NV)
        self.predict_pos = ti.Vector.field(3, float, NV)
        self.old_pos = ti.Vector.field(3, float, NV)
        self.vel = ti.Vector.field(3, float, NV)  # velocity of particles
        self.mass = ti.field(float, NV)  # mass of particles
        self.inv_mass = ti.field(float, NV)  # inverse mass of particles
        self.tet_indices = ti.Vector.field(4, int, NT)
        self.B = ti.Matrix.field(3, 3, float, NT)  # D_m^{-1}
        self.lagrangian = ti.field(float, NT)  # lagrangian multipliers
        self.rest_volume = ti.field(float, NT)  # rest volume of each tet
        self.inv_V = ti.field(float, NT)  # inverse volume of each tet
        self.alpha_tilde = ti.field(float, NT)

        self.par_2_tet = ti.field(int, NV)
        self.gradC = ti.Vector.field(3, ti.f32, shape=(NT, 4))
        self.constraints = ti.field(ti.f32, shape=(NT))
        self.dpos = ti.Vector.field(3, ti.f32, shape=(NV))
        self.residual = ti.field(ti.f32, shape=NT)
        self.dual_residual = ti.field(ti.f32, shape=NT)
        self.dlambda = ti.field(ti.f32, shape=NT)
        self.tet_centroid = ti.Vector.field(3, ti.f32, shape=NT)
        self.potential_energy = ti.field(ti.f32, shape=())
        self.inertial_energy = ti.field(ti.f32, shape=())
        self.ele = self.tet_indices

    def initialize(self):
        info(f"Initializing mesh")

        # read models
        self.model_pos = self.model_pos.astype(np.float32)
        self.model_tet = self.model_tet.astype(np.int32)
        self.pos.from_numpy(self.model_pos)
        self.tet_indices.from_numpy(self.model_tet)

        inv_mu = 1.0 / args.mu
        inv_h2 = 1.0 / args.delta_t / args.delta_t
        # init inv_mass rest volume alpha_tilde etc.
        init_physics(
            self.pos,
            self.old_pos,
            self.vel,
            self.tet_indices,
            self.B,
            self.rest_volume,
            self.inv_V,
            self.mass,
            self.inv_mass,
            self.alpha_tilde,
            self.par_2_tet,
            inv_mu,
            inv_h2,
        )
        self.alpha_tilde_np = self.alpha_tilde.to_numpy()


    def reinit(self):
        # args.reinit = "squash"
        # FIXME: no reinit will cause bug, why? FIXED: because when there is no deformation, the gradient will be in any direction! Sigma=(1,1,1) There will be singularity issue! We need to jump the constraint=0 case.
        # reinit pos
        self.initial_pos = self.pos.to_numpy()
        if args.reinit == "random":
            random_val = np.random.rand(self.pos.shape[0], 3)
            self.pos.from_numpy(random_val)
        elif args.reinit == "enlarge":
            self.pos.from_numpy(self.model_pos * 1.5)
        elif args.reinit == "squash":
            self.pos.from_numpy(self.model_pos * 0.01)
        elif args.reinit == "freefall":
            args.gravity = [0, -9.8, 0]
            args.use_ground_collision = 1
 
            #for ground collision response
            min_pos_y = np.min(self.pos.to_numpy()[:,1])
            max_pos_y = np.max(self.pos.to_numpy()[:,1])
            if min_pos_y < 0.0:
                # move the model above the ground
                logging.warning(f"move the model above the ground")
                self.pos.from_numpy(self.pos.to_numpy() + np.array([0, -min_pos_y+(max_pos_y-min_pos_y)*0.01, 0]))
            self.ground_pos = 0.0


    # calc_dual use the base class's

    def calc_primal(self):
        def calc_primal_imply(self,G,M_inv):
            MASS = scipy.sparse.diags(1.0/(M_inv.diagonal()), format="csr")
            primary_residual = MASS @ (self.pos.to_numpy().flatten() - self.predict_pos.to_numpy().flatten()) - G.transpose() @ self.lagrangian.to_numpy()
            where_zeros = np.where(M_inv.diagonal()==0)
            primary_residual = np.delete(primary_residual, where_zeros)
            return primary_residual
        G = fill_G()
        primary_residual = self.calc_primal_imply(G, self.M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.linalg.norm(np.concatenate((self.dual_residual.to_numpy(), primary_residual))).astype(float)
        return primal_r, Newton_r


    def calc_strain(self)->float:
        """ 
         The strain of a tet is exactly the ARAP constraint of the tet
         S = diag(sigma1, sigma2, sigma3)
         constraint = sqrt((sigma1-1)^2 + (sigma2-1)^2 + (sigma3-1)^2)
        """
        self.strain = self.constraints.to_numpy()
        self.max_strain = np.max(self.strain)
        return self.max_strain
    
    def update_constraints(self):
        update_constraints_kernel(self.pos, self.tet_indices, self.B, self.constraints)

    # calc_energy use the base class's 
    
    def compute_C_and_gradC(self):
        compute_C_and_gradC_kernel(self.pos_mid, self.tet_indices, self.B, self.constraints, self.gradC)

    @timeit
    def dlam2dpos(self,dlam):
        self.dlambda.from_numpy(dlam)
        self.dpos.fill(0.0)
        dlam2dpos_kernel(self.gradC, self.tet_indices, self.inv_mass, self.dlambda, self.lagrangian, self.dpos)
        self.update_pos()

    def compute_b(self):
        b = -self.constraints.to_numpy() - self.alpha_tilde_np * self.lagrangian.to_numpy()
        return b
    
    def project_arap_xpbd(self):
        project_arap_xpbd_kernel(
            self.pos_mid,
            self.tet_indices,
            self.inv_mass,
            self.lagrangian,
            self.B,
            self.pos,
            self.alpha_tilde,
            self.constraints,
            self.residual,
            self.gradC,
            self.dlambda,
            self.dpos,
            args.omega
        )


    @timeit
    def solveSoft_cuda(self):
        arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
        arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
        arr2d_float = ctl.ndpointer(dtype=np.float32, ndim=2, flags='aligned, c_contiguous')
        arr2d_int = ctl.ndpointer(dtype=np.int32, ndim=2, flags='aligned, c_contiguous')
        arr3d_float = ctl.ndpointer(dtype=np.float32, ndim=3, flags='aligned, c_contiguous')
        arr2i_float = ctl.ndpointer(dtype=np.int32, ndim=2, flags='aligned, c_contiguous')
        c_size_t = ctypes.c_size_t
        c_float = ctypes.c_float
        c_int = ctypes.c_int


        extlib.solveSoft_set_data.argtypes = [
            c_size_t,
            c_size_t,
            arr2d_float, #pos
            arr_float, #alpha_tilde
            arr_float, #rest_len
            arr2i_float, #vert
            arr_float, #mass
            c_float, #delta_t
            arr3d_float, #restmatrix
            arr_float, #lambda
        ]
        extlib.solveSoft_get_data.argtypes = [
            arr_float, #dlambda
            arr2d_float, #dpos
            arr_float, #constraints
            arr3d_float, #gradC
            arr_float, #b_out
        ]
        extlib.solveSoft_get_data.restype = c_float
        extlib.solveSoft_run.argtypes = [
            arr2d_float, #pos
        ]
        extlib.solveSoft_run.restype = c_float



        extlib.solveSoft_set_data(
            self.NV,
            self.NCONS,
            self.pos.to_numpy(),
            self.alpha_tilde.to_numpy(),
            self.rest_volume.to_numpy(),
            self.tet_indices.to_numpy(),
            self.inv_mass.to_numpy(),
            args.delta_t,
            self.B.to_numpy(),
            self.lagrangian.to_numpy(),
        )

        pos_out = self.pos.to_numpy()
        extlib.solveSoft_run(pos_out)


        self.dlambda_out = np.zeros(self.NCONS, dtype=np.float32)
        self.dpos_out = np.zeros((self.NV, 3), dtype=np.float32)
        self.constraints_out = np.zeros(self.NCONS, dtype=np.float32)
        self.gradC_out = np.zeros((self.NCONS, 4, 3), dtype=np.float32)
        self.b_out = np.zeros(self.NCONS, dtype=np.float32)
        extlib.solveSoft_get_data(
            self.dlambda_out, 
            self.dpos_out,
            self.constraints_out,
            self.gradC_out,
            self.b_out,
        )

        self.constraints.from_numpy(self.constraints_out)
        self.gradC.from_numpy(self.gradC_out.copy())
        self.b = self.b_out



        dlam, self.r_iter.r_Axb = self.linsol.run(self.b)
        self.dlam2dpos(dlam)

    @timeit
    def solveSoft_python(self):
        self.pos_mid.from_numpy(self.pos.to_numpy())
        self.compute_C_and_gradC()
        self.b = self.compute_b()
        x, self.r_iter.r_Axb = self.linsol.run(self.b)
        self.dlam2dpos(x)

    def solveSoft(self):
        self.solveSoft_python()

    def do_pre_iter0(self):
        self.update_constraints() # for calculation of r0
        self.r_iter.calc_r0()

    def substep_all_solver(self):
        self.semi_euler()
        self.lagrangian.fill(0)
        self.do_pre_iter0()
        for self.ite in range(args.maxiter):
            self.r_iter.tic_iter = perf_counter()
            self.solveSoft()
            do_post_iter(self, get_A0_cuda)
            if self.r_iter.check():
                break
        self.collision_response()
        self.n_outer_all.append(self.ite+1)
        self.update_vel()
        if args.use_pintoanimation:
            self.read_geo_pinpos()

    def substep_xpbd(self):
        self.semi_euler()
        self.lagrangian.fill(0)
        self.do_pre_iter0()
        for self.ite in range(args.maxiter):
            self.r_iter.tic_iter = perf_counter()
            self.project_arap_xpbd()
            self.do_post_iter_xpbd()
            if self.r_iter.check():
                break
        self.collision_response()
        self.n_outer_all.append(self.ite+1)
        self.update_vel()



# ---------------------------------------------------------------------------- #
#                                    kernels                                   #
# ---------------------------------------------------------------------------- #


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
def set_pinpos_kernel(pin:ti.types.ndarray(), pos:ti.template(), pinpos:ti.types.ndarray()):
    for i in range(pos.shape[0]):
        if pin[i]:
            pos[i] = pinpos[i]
            

@ti.kernel
def init_B(
    pos: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
):
    for i in range(tet_indices.shape[0]):
        ia, ib, ic, id = tet_indices[i]
        p0, p1, p2, p3 = pos[ia], pos[ib], pos[ic], pos[id]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        B[i] = D_m.inverse()



@ti.kernel
def init_alpha_tilde(
    pos: ti.template(),
    tet_indices: ti.template(),
    rest_volume: ti.template(),
    alpha_tilde: ti.template(),
    inv_mu: ti.f32,
    inv_h2: ti.f32,
):
    for i in range(tet_indices.shape[0]):
        ia, ib, ic, id = tet_indices[i]
        p0, p1, p2, p3 = pos[ia], pos[ib], pos[ic], pos[id]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        rest_volume[i] = 1.0 / 6.0 * ti.abs(D_m.determinant())
        alpha_tilde[i] = inv_h2 * inv_mu  / rest_volume[i]


@ti.kernel
def init_physics(
    pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    rest_volume: ti.template(),
    inv_V: ti.template(),
    mass: ti.template(),
    inv_mass: ti.template(),
    alpha_tilde: ti.template(),
    par_2_tet: ti.template(),
    inv_mu: ti.f32,
    inv_h2: ti.f32,
):
    # init pos, old_pos, vel
    for i in range(pos.shape[0]):
        old_pos[i] = pos[i]
        vel[i] = ti.Vector([0, 0, 0])

    # init B and rest_volume
    total_volume = 0.0
    for i in range(tet_indices.shape[0]):
        ia, ib, ic, id = tet_indices[i]
        p0, p1, p2, p3 = pos[ia], pos[ib], pos[ic], pos[id]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        B[i] = D_m.inverse()

        rest_volume[i] = 1.0 / 6.0 * ti.abs(D_m.determinant())
        inv_V[i] = 1.0 / rest_volume[i]
        total_volume += rest_volume[i]

    # init mass
    # for i in tet_indices:
    #     ia, ib, ic, id = tet_indices[i]
    #     mass_density = args.total_mass / total_volume
    #     tet_mass = mass_density * rest_volume[i]
    #     avg_mass = tet_mass / 4.0
    #     mass[ia] += avg_mass
    #     mass[ib] += avg_mass
    #     mass[ic] += avg_mass
    #     mass[id] += avg_mass
    for i in inv_mass:
        inv_mass[i] = 1.0 

    for i in alpha_tilde:
        alpha_tilde[i] = inv_h2 * inv_mu * inv_V[i]

    # init par_2_tet
    for i in tet_indices:
        ia, ib, ic, id = tet_indices[i]
        par_2_tet[ia], par_2_tet[ib], par_2_tet[ic], par_2_tet[id] = i, i, i, i




@ti.func
def make_matrix(x, y, z):
    return ti.Matrix(
        [
            [x, 0, 0, y, 0, 0, z, 0, 0],
            [0, x, 0, 0, y, 0, 0, z, 0],
            [0, 0, x, 0, 0, y, 0, 0, z],
        ]
    )


@ti.func
def compute_gradient(U, S, V, B):
    sum_sigma = sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)

    # (dcdS00, dcdS11, dcdS22)
    dcdS = 1.0 / sum_sigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1, S[2, 2] - 1])
    # Compute (dFdx)^T
    dFdp1T = make_matrix(B[0, 0], B[0, 1], B[0, 2])
    dFdp2T = make_matrix(B[1, 0], B[1, 1], B[1, 2])
    dFdp3T = make_matrix(B[2, 0], B[2, 1], B[2, 2])
    # Compute (dsdF)
    u00, u01, u02 = U[0, 0], U[0, 1], U[0, 2]
    u10, u11, u12 = U[1, 0], U[1, 1], U[1, 2]
    u20, u21, u22 = U[2, 0], U[2, 1], U[2, 2]
    v00, v01, v02 = V[0, 0], V[0, 1], V[0, 2]
    v10, v11, v12 = V[1, 0], V[1, 1], V[1, 2]
    v20, v21, v22 = V[2, 0], V[2, 1], V[2, 2]
    dsdF00 = ti.Vector([u00 * v00, u01 * v01, u02 * v02])
    dsdF10 = ti.Vector([u10 * v00, u11 * v01, u12 * v02])
    dsdF20 = ti.Vector([u20 * v00, u21 * v01, u22 * v02])
    dsdF01 = ti.Vector([u00 * v10, u01 * v11, u02 * v12])
    dsdF11 = ti.Vector([u10 * v10, u11 * v11, u12 * v12])
    dsdF21 = ti.Vector([u20 * v10, u21 * v11, u22 * v12])
    dsdF02 = ti.Vector([u00 * v20, u01 * v21, u02 * v22])
    dsdF12 = ti.Vector([u10 * v20, u11 * v21, u12 * v22])
    dsdF22 = ti.Vector([u20 * v20, u21 * v21, u22 * v22])

    # Compute (dcdF)
    dcdF = ti.Vector(
        [
            dsdF00.dot(dcdS),
            dsdF10.dot(dcdS),
            dsdF20.dot(dcdS),
            dsdF01.dot(dcdS),
            dsdF11.dot(dcdS),
            dsdF21.dot(dcdS),
            dsdF02.dot(dcdS),
            dsdF12.dot(dcdS),
            dsdF22.dot(dcdS),
        ]
    )
    g1 = dFdp1T @ dcdF
    g2 = dFdp2T @ dcdF
    g3 = dFdp3T @ dcdF
    g0 = -g1 - g2 - g3
    return g0, g1, g2, g3



@ti.kernel
def compute_C_and_gradC_kernel(
    pos: ti.template(),
    tet_indices: ti.template(),
    B: ti.template(),
    constraints: ti.template(),
    gradC: ti.template(),
):
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]
        x0, x1, x2, x3 = pos[p0], pos[p1], pos[p2], pos[p3]
        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        # U, S, V = np.linalg.svd(F.to_numpy())
        constraints[t] = sqrt((S[0,0] - 1) ** 2 + (S[1,1] - 1) ** 2 + (S[2,2] - 1) ** 2)
        if constraints[t]>1e-6: #CAUTION! When the constraint is too small, there is no deformation at all, the gradient will be in any direction! Sigma=(1,1,1) There will be singularity issue!
            gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = compute_gradient(U, S, V, B[t])




@ti.kernel
def update_constraints_kernel(
    pos: ti.template(), #pos not pos_mid
    tet_indices: ti.template(),
    B: ti.template(),
    constraints: ti.template(),
):
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]
        x0, x1, x2, x3 = pos[p0], pos[p1], pos[p2], pos[p3]
        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraints[t] = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)




@ti.kernel
def project_arap_xpbd_kernel(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    inv_mass: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    pos: ti.template(),
    alpha_tilde: ti.template(),
    constraint: ti.template(),
    residual: ti.template(),
    gradC: ti.template(),
    dlambda: ti.template(),
    dpos: ti.template(),
    omega: ti.f32
):

    # ti.loop_config(serialize=meta.serialize)
    for i in pos:
        pos_mid[i] = pos[i]

    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]

        x0, x1, x2, x3 = pos_mid[p0], pos_mid[p1], pos_mid[p2], pos_mid[p3]

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint[t] = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        if constraint[t] > 1e-6:
            g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
            gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0, g1, g2, g3
            denominator = (
                inv_mass[p0] * g0.norm_sqr()
                + inv_mass[p1] * g1.norm_sqr()
                + inv_mass[p2] * g2.norm_sqr()
                + inv_mass[p3] * g3.norm_sqr()
            )
            residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t])
            dlambda[t] = residual[t] / (denominator + alpha_tilde[t])
            lagrangian[t] += dlambda[t]
    for t in range(tet_indices.shape[0]):
        if constraint[t] > 1e-6:
            p0 = tet_indices[t][0]
            p1 = tet_indices[t][1]
            p2 = tet_indices[t][2]
            p3 = tet_indices[t][3]
            pos[p0] += omega * inv_mass[p0] * dlambda[t] * gradC[t, 0]
            pos[p1] += omega * inv_mass[p1] * dlambda[t] * gradC[t, 1]
            pos[p2] += omega * inv_mass[p2] * dlambda[t] * gradC[t, 2]
            pos[p3] += omega * inv_mass[p3] * dlambda[t] * gradC[t, 3]
            dpos[p0] += omega * inv_mass[p0] * dlambda[t] * gradC[t, 0]
            dpos[p1] += omega * inv_mass[p1] * dlambda[t] * gradC[t, 1]
            dpos[p2] += omega * inv_mass[p2] * dlambda[t] * gradC[t, 2]
            dpos[p3] += omega * inv_mass[p3] * dlambda[t] * gradC[t, 3]







@ti.kernel
def dlam2dpos_kernel(gradC:ti.template(),
                     tet_indices:ti.template(),
                     inv_mass:ti.template(),
                     dlambda:ti.template(),
                     lagrangian:ti.template(),
                     dpos:ti.template()

):
    for i in range(tet_indices.shape[0]):
        idx0, idx1, idx2, idx3 = tet_indices[i]
        lagrangian[i] += dlambda[i]
        dpos[idx0] += inv_mass[idx0] * dlambda[i] * gradC[i, 0]
        dpos[idx1] += inv_mass[idx1] * dlambda[i] * gradC[i, 1]
        dpos[idx2] += inv_mass[idx2] * dlambda[i] * gradC[i, 2]
        dpos[idx3] += inv_mass[idx3] * dlambda[i] * gradC[i, 3]

# ---------------------------------------------------------------------------- #
#                                    fill A                                    #
# ---------------------------------------------------------------------------- #

def fill_G(ist):
    ii, jj, vv = np.zeros(ist.NT*ist.MAX_ADJ, dtype=np.int32), np.zeros(ist.NT*ist.MAX_ADJ, dtype=np.int32), np.zeros(ist.NT*ist.MAX_ADJ, dtype=np.float32)
    fill_gradC_triplets_kernel(ii,jj,vv, ist.gradC, ist.tet_indices)
    G = scipy.sparse.coo_array((vv, (ii, jj)))
    return G


# TODO: DEPRECATE
def fill_A_by_spmm(ist,  M_inv, ALPHA):
    ii, jj, vv = np.zeros(ist.NT*ist.MAX_ADJ, dtype=np.int32), np.zeros(ist.NT*ist.MAX_ADJ, dtype=np.int32), np.zeros(ist.NT*ist.MAX_ADJ, dtype=np.float32)
    fill_gradC_triplets_kernel(ii,jj,vv, ist.gradC, ist.tet_indices)
    G = scipy.sparse.coo_array((vv, (ii, jj)))

    # assemble A
    A = G @ M_inv @ G.transpose() + ALPHA
    A = scipy.sparse.csr_matrix(A, dtype=np.float32)
    # A = scipy.sparse.diags(A.diagonal(), format="csr")
    return A



def should_setup():
    return ((ist.frame%args.setup_interval==0 or (args.restart==True and ist.frame==args.restart_frame)) and (ist.ite==0))



def fill_A_csr_ti(ist):
    fill_A_csr_lessmem_kernel(ist.spmat_data, ist.spmat_indptr, ist.ii, ist.jj, ist.nnz, ist.alpha_tilde, ist.inv_mass, ist.gradC, ist.tet_indices)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(ist.NT, ist.NT))
    return A


# 求两个长度为4的数组的交集
@ti.func
def intersect(a, b):   
    # a,b: 4个顶点的id, e:当前ele的id
    k=0 # 第几个共享的顶点， 0, 1, 2, 3
    c = ti.Vector([-1,-1,-1])         # 共享的顶点id存在c中
    order = ti.Vector([-1,-1,-1])     # 共享的顶点是当前ele的第几个顶点
    order2 = ti.Vector([-1,-1,-1])    # 共享的顶点是邻接ele的第几个顶点
    for i in ti.static(range(4)):     # i:当前ele的第i个顶点
        for j in ti.static(range(4)): # j:邻接ele的第j个顶点
            if a[i] == b[j]:
                c[k] = a[i]         
                order[k] = i          
                order2[k] = j
                k += 1
    return k, c, order, order2

# for cnt version, require init_A_CSR_pattern() to be called first
@ti.kernel
def fill_A_csr_lessmem_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                      indptr:ti.types.ndarray(dtype=ti.i32), 
                      ii:ti.types.ndarray(dtype=ti.i32), 
                      jj:ti.types.ndarray(dtype=ti.i32),
                      nnz:ti.i32,
                      alpha_tilde:ti.template(),
                      inv_mass:ti.template(),
                      gradC:ti.template(),
                      ele: ti.template()
                    ):
    for n in range(nnz):
        i = ii[n] # row index,  current element id
        j = jj[n] # col index,  adjacent element id, adj_id
        k = n - indptr[i] # k: 第几个非零元
        if i == j: # diag
            m1,m2,m3,m4 = inv_mass[ele[i][0]], inv_mass[ele[i][1]], inv_mass[ele[i][2]], inv_mass[ele[i][3]]
            g1,g2,g3,g4 = gradC[i,0], gradC[i,1], gradC[i,2], gradC[i,3]
            data[n] = m1*g1.norm_sqr() + m2*g2.norm_sqr() + m3*g3.norm_sqr() + m4*g4.norm_sqr() + alpha_tilde[i]
            continue
        offdiag=0.0
        n_shared_v, shared_v, shared_v_order_in_cur, shared_v_order_in_adj = intersect(ele[i], ele[j])
        for kv in range(n_shared_v): #kv 第几个共享点
            o1 = shared_v_order_in_cur[kv]
            o2 = shared_v_order_in_adj[kv]
            sv = shared_v[kv]  #sv: 共享的顶点id    shared vertex
            sm = inv_mass[sv]      #sm: 共享的顶点的质量倒数 shared inv mass
            offdiag += sm*gradC[i,o1].dot(gradC[j,o2])
        data[n] = offdiag


# for cnt version, require init_A_CSR_pattern() to be called first
# legacy version, now we use less memory version
# fill_A_csr_kernel(ist.spmat_data, ist.spmat_indptr, ist.ii, ist.jj, ist.nnz, ist.alpha_tilde, ist.inv_mass, ist.gradC, ist.tet_indices, ist.n_shared_v, ist.shared_v, ist.shared_v_order_in_cur, ist.shared_v_order_in_adj)
@ti.kernel
def fill_A_csr_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                      indptr:ti.types.ndarray(dtype=ti.i32), 
                      ii:ti.types.ndarray(dtype=ti.i32), 
                      jj:ti.types.ndarray(dtype=ti.i32),
                      nnz:ti.i32,
                      alpha_tilde:ti.template(),
                      inv_mass:ti.template(),
                      gradC:ti.template(),
                      ele: ti.template(),
                      n_shared_v:ti.types.ndarray(),
                      shared_v:ti.types.ndarray(),
                      shared_v_order_in_cur:ti.types.ndarray(),
                      shared_v_order_in_adj:ti.types.ndarray(),
                    ):
    for n in range(nnz):
        i = ii[n] # row index,  current element id
        j = jj[n] # col index,  adjacent element id, adj_id
        k = n - indptr[i] # k: 第几个非零元
        if i == j: # diag
            m1,m2,m3,m4 = inv_mass[ele[i][0]], inv_mass[ele[i][1]], inv_mass[ele[i][2]], inv_mass[ele[i][3]]
            g1,g2,g3,g4 = gradC[i,0], gradC[i,1], gradC[i,2], gradC[i,3]
            data[n] = m1*g1.norm_sqr() + m2*g2.norm_sqr() + m3*g3.norm_sqr() + m4*g4.norm_sqr() + alpha_tilde[i]
            continue
        offdiag=0.0
        for kv in range(n_shared_v[i, k]): #kv 第几个共享点
            o1 = shared_v_order_in_cur[i,k,kv]
            o2 = shared_v_order_in_adj[i,k,kv]
            sv = shared_v[i,k,kv]  #sv: 共享的顶点id    shared vertex
            sm = inv_mass[sv]      #sm: 共享的顶点的质量倒数 shared inv mass
            offdiag += sm*gradC[i,o1].dot(gradC[j,o2])
        data[n] = offdiag





# ---------------------------------------------------------------------------- #
#                              end fill A                                      #
# ---------------------------------------------------------------------------- #
def AMG_A():
    tic2 = perf_counter()
    extlib.fastFillSoft_run(ist.pos.to_numpy(), ist.gradC.to_numpy())
    extlib.fastmg_set_A0_from_fastFillSoft()
    logging.info(f"    fill_A time: {(perf_counter()-tic2)*1000:.0f}ms")


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

def get_A0_python()->scipy.sparse.csr_matrix:
    A = fill_A_csr_ti(ist)
    return A

def get_A0_cuda()->scipy.sparse.csr_matrix:
    AMG_A()
    A = fetch_A_from_cuda(0)
    return A


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def init_linear_solver():
    if args.solver_type == "AMG":
        from engine.soft.graph_coloring import graph_coloring_v2
        def gc():
            return graph_coloring_v2(fetch_A_from_cuda, ist.num_levels, extlib, ist.model_path)
        
        if args.use_cuda:
            linsol = AmgCuda(
                args=args,
                extlib=extlib,
                get_A0=get_A0_cuda,
                should_setup=should_setup,
                fill_A_in_cuda=AMG_A,
                graph_coloring=gc,
                copy_A=True,
            )
        else:
            linsol = AmgPython(args, get_A0_python, should_setup)
    elif args.solver_type == "AMGX":
        linsol = AmgxSolver(args.amgx_config, get_A0_python, args.cuda_dir, args.amgx_lib_dir)
    elif args.solver_type == "DIRECT":
        linsol = DirectSolver(get_A0_python)
    elif args.solver_type == "XPBD":
        linsol=None
    else:
        linsol=None
    return linsol


def init():
    tic = perf_counter()
    process_dirs(args)
    init_logger(args)
    global extlib
    extlib = init_extlib(args,sim="soft")
    global ist
    ist = SoftBody(args.model_path)
    print(f"initialize time:", perf_counter()-tic)


def main():
    init()
    from engine.util import main_loop
    main_loop(ist,args)

if __name__ == "__main__":
    main()
