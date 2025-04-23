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
from engine.mesh_io import write_mesh, read_tet, read_geo
from engine.common_args import add_common_args, parse_json_args
from engine.init_extlib import init_extlib
from engine.solver.amg_python import AmgPython
from engine.solver.amg_cuda import AmgCuda
from engine.solver.amgx_solver import AmgxSolver
from engine.solver.direct_solver import DirectSolver
from engine.util import calc_norm,  ResidualDataOneIter, init_logger, timeit, python_list_to_ti_field
from engine.util import vec_is_equal
from engine.physical_base import PhysicalBase
from script.convert.geo import Geo
from engine.energy import compute_energy

def init_args():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("-mu", type=float, default=1e6)
    parser.add_argument("-damping_coeff", type=float, default=1.0)
    parser.add_argument("-model_path", type=str, default=f"data/model/bunny_small/bunny_small.node")
    # "data/model/cube/minicube.node"
    # "data/model/bunny1k2k/coarse.node"
    # "data/model/bunny_small/bunny_small.node"
    # "data/model/bunnyBig/bunnyBig.node"
    # "data/model/bunny85w/bunny85w.node"
    # "data/model/ball/ball.node"
    parser.add_argument("-reinit", type=str, default="enlarge",choices=["random","enlarge","squash","freefall","beam","twist_bar"])
    parser.add_argument("-large", action="store_true")
    parser.add_argument("-small", action="store_true")
    parser.add_argument("-omega", type=float, default=0.1)
    parser.add_argument("-smoother_type", type=str, default="jacobi")
    parser.add_argument("-use_line_search", type=int, default=True)


    args = parser.parse_args()


    if args.use_json and args.json_path:
        args = parse_json_args(args, args.json_path)

    if args.large:
        args.model_path = f"data/model/bunny85w/bunny85w.node"
    if args.small:
        args.model_path = f"data/model/bunny1k2k/coarse.node"

    if args.arch == "gpu":
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)

    return args

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
        self.damping_coeff = args.damping_coeff


        dir = str(Path(mesh_file).parent.stem)
        self.sim_name = f"soft3d-{dir}-{str(Path(mesh_file).stem)}"
        self.frame=args.start_frame
        self.initial_frame=args.start_frame

        if args.use_gravity:
            self.gravity = ti.Vector([0.0, -9.8, 0])
        else:
            self.gravity = ti.Vector([0.0, 0.0, 0.0])

        if args.use_extra_spring or args.use_pintoanimation or args.use_pintotarget or args.use_muscle2muscle:
            args.use_houdini_data=1

        if args.use_houdini_data:
            self.read_geo_rest()
            if args.use_extra_spring:
                self.read_extra_spring_rest()
            if args.use_pintotarget:
                self.read_pintotarget_rest()
            if args.use_muscle2muscle:
                self.read_muscle2muscle_rest()
            self.init_physics()
        else:
            self.build_mesh(mesh_file)
            self.NCONS = self.NT
            self.allocate_fields(self.NV, self.NT)
            self.initialize()
            if args.export_mesh:
                write_mesh(args.out_dir + f"/mesh/{0:04d}", self.pos.to_numpy(), self.model_tri)
        self.force = np.zeros((self.NV, 3), dtype=np.float32)

        if args.calc_rbm:
            # CAUTION: THIS IS ONLY WORK for the primal system with 3nx3n matrix, otherwise the shape of B will be wrong!
            self.rbm = get_rbm(self.initial_pos)
            np.save(f"rbm.npy", self.rbm)
        info(f"Creating instance done")

    
    def load(self, filename):
        if Path(filename).suffix == ".txt":
            pos = load_pos_from_txt(filename)
            self.pos.from_numpy(pos)
            self.vel.fill(0)
        elif Path(filename).suffix == ".node":
            pos = load_pos_from_node(filename)
            self.pos.from_numpy(pos)
            self.vel.fill(0)
        else:
            logging.warning("unknown file format")
        print(f"loaded pos from {filename}")


    def line_search(self, x, dpos, ls_beta=0.5, EPSILON=1e-9,):
        """
        x: position of vertices, shape=(NV,3), numpy array
        dpos: dpos, shape=(NV,3), numpy array
        reutrn: step size
        """

        if not args.use_line_search:
            return self.omega
        
        def evalutate_objective(x):
            return calc_dualnorm_kernel(x, self.tet_indices,self.lagrangian,self.B,self.alpha_tilde,)

        t = 1.0/ls_beta
        ls_times = 0
        currentObjectiveValue = evalutate_objective(x)
        while ls_times==0 or (lhs >= rhs and t > EPSILON):
            t *= ls_beta
            x_plus_tdx = x + t*dpos
            lhs = evalutate_objective(x_plus_tdx)
            rhs = currentObjectiveValue 
            ls_times += 1
        obj = lhs
        # print(f'    obj: {obj:.8e}')
        # print(f'    ls_times: {ls_times}')
        # print(f'    step size: {t}')

        if t < EPSILON:
            t = 0.0
        return t


    def read_extra_spring_rest(self,):
        dir = prj_path + "/" + args.geo_dir + "/"
        consgeo = Geo(dir+f"cons_{self.initial_frame}.geo")
        self.consgeo_rest = consgeo
        
        # read connectivity
        # first column is target(driving point), second column is source(driven)
        pts1 = np.array(consgeo.get_pts())
        pts = ti.field(int, pts1.shape[0])
        pts.from_numpy(pts1)

        # read sim pos(to be driven)
        pos1 = np.array(self.geo_rest.get_pos(),dtype=np.float32)
        pos = ti.Vector.field(3, ti.f32, pos1.shape[0])
        pos.from_numpy(pos1)

        # read target pos(driving)
        tp = consgeo.get_target_pos()
        self.target_pos = python_list_to_ti_field(tp)

        from engine.constraints.distance_constraints import DistanceConstraintsAttach
        self.extra_springs = DistanceConstraintsAttach(pts, pos, self.target_pos)

        # optional data(inv_mass, stiffness, restlength)
        self.extra_springs.set_alpha(consgeo.get_stiffness())
        self.extra_springs.set_rest_len(consgeo.get_restlength())


    def read_muscle2muscle_rest(self,):
        """ read muscle2muscle topology from geo file
            It start from pt_index 1
        """
        dir = prj_path + "/" + args.geo_dir + "/"
        m2mgeo = Geo(dir+f"m2m.geo")
        self.m2mgeo = m2mgeo

        # source pt(interior pt of muscle A)
        src = np.array(m2mgeo.get_pts())

        # target pts (surface pts of muscle B, could be multiple)
        tps = (m2mgeo.get_target_pts())

        # pairs of (p1, p2)
        pairs = []
        for i,p1 in enumerate(src):
            p2s = tps[i]
            for k,p2 in enumerate(p2s):
                pairs.append((p1,p2))

        pairs_np = np.array(pairs)
        p1 = python_list_to_ti_field(pairs_np[:,0].tolist())
        p2 = python_list_to_ti_field(pairs_np[:,1].tolist())
        from engine.constraints.distance_constraints import DistanceConstraints
        self.m2mCons = DistanceConstraints(p1,p2, self.pos )


    def read_pintotarget_rest(self,):
        dir = prj_path + "/" + args.geo_dir + "/"
        consgeo = Geo(dir+f"cons_{self.initial_frame}.geo")
        self.consgeo_rest = consgeo
        
        # read connectivity
        # target_pos is driving point, pts is source points(to be driven)
        pts1 = np.array(consgeo.get_pts())
        pts = ti.field(int, pts1.shape[0])
        pts.from_numpy(pts1)

        # read sim pos(to be driven)
        pos1 = np.array(self.geo_rest.get_pos(),dtype=np.float32)
        pos = ti.Vector.field(3, ti.f32, pos1.shape[0])
        pos.from_numpy(pos1)

        # read target pos(driving)
        target_pos = np.array(consgeo.get_target_pos(),dtype=np.float32)
        self.target_pos = ti.Vector.field(3, ti.f32, target_pos.shape[0])
        self.target_pos.from_numpy(target_pos)

        from engine.constraints.distance_constraints import PinToTarget
        self.pintotarget = PinToTarget(pts, pos, self.target_pos)


    @timeit
    def read_target_pos(self):
        dir = prj_path + "/" + args.geo_dir + "/"
        geo = Geo(dir+f"cons_{self.frame}.geo")
        tp = np.array(geo.get_target_pos(),dtype=np.float32)
        self.target_pos.from_numpy(np.array(tp, dtype=np.float32))
        ...


    def read_geo_pinpos(self):
        dir = prj_path + "/" + args.geo_dir + "/"
        geo = Geo(dir+f"physdata_{ist.frame}.geo")
        pinpos = np.array(geo.get_pos())
        assert pinpos.shape[0] == self.pos.shape[0]
        # set_pinpos_kernel(self.pin, self.pos, pinpos)
        
        self.pinlist = np.where(self.pin)[0]
        self.inv_mass_np = self.inv_mass.to_numpy()
        self.inv_mass_np[self.pinlist] = 0.0
        self.inv_mass.from_numpy(self.inv_mass_np)

        pos_ = self.pos.to_numpy()
        pos_[self.pin] = pinpos[self.pin]
        self.pos.from_numpy(pos_)


    def read_geo_mesh(self,filename):
        geo = Geo(filename)
        vert = np.array(geo.get_vert(),dtype=np.int32)
        pos = np.array(geo.get_pos(), dtype=np.float32)
        self.NV = pos.shape[0]
        self.NT = vert.shape[0]
        self.NCONS = self.NT
        self.allocate_fields(self.NV, self.NT)

        self.vert = vert
        self.pos.from_numpy(pos)
        self.pos_mid.from_numpy(pos)
        self.old_pos.from_numpy(pos)
        self.tet_indices.from_numpy(vert)
        self.geodir = dir
        self.geo = geo
        self.geo_rest = geo


    def read_geo_rest(self):
        dir = prj_path + "/" + args.geo_dir + "/"
        if os.path.exists(dir+"restpos.geo"):
            geo = Geo(dir+"restpos.geo")
        elif os.path.exists(dir+"physdata_1.geo"):
            geo = Geo(dir+"physdata_1.geo")
        else:
            raise FileNotFoundError(f"restpos.geo or physdata_1.geo not found in {dir}")
         
        pin = np.array(geo.get_gluetoaniamtion(),dtype=np.bool_)
        vert = np.array(geo.get_vert(),dtype=np.int32)
        pinpos = np.array(geo.get_pos(), dtype=np.float32)

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
        self.geo_rest = geo
        
        # read mass from geo
        im = np.array(geo.get_mass(), dtype=np.float32)
        im = 1.0 / im[np.isnan(im)==False]
        # set pinned point inv_mass to 0
        im[pin] = 0.0

        self.inv_mass.from_numpy(im)


    def init_physics(self):
        init_B(self.pos, self.tet_indices, self.B)
        init_alpha_tilde(self.pos, self.tet_indices, self.rest_volume, self.alpha_tilde, 1.0 / args.mu, 1.0 / args.delta_t / args.delta_t)
        self.alpha_tilde_np = self.alpha_tilde.to_numpy()

    def write_geo(self, output=None):
        self.geo.set_positions(self.pos.to_numpy())
        if output is None:
            output = self.geodir+f"physdata_{self.frame}_out.geo"
        self.geo.write(output)

    def build_mesh(self,mesh_file):
        tic = time.perf_counter()
        if Path(mesh_file).suffix == ".node":
            self.model_pos, self.model_tet, self.model_tri = read_tet(mesh_file, build_face_flag=True)
        elif Path(mesh_file).suffix == ".geo":
            self.model_pos, self.model_tet, self.model_tri, self.geo = read_geo(mesh_file, build_face_flag=True)
        print(f"Tetrahedrons:{self.model_tet.shape[0]}, Vertices:{self.model_pos.shape[0]}")
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
        self.lagrangian_D = ti.field(float, NT)  # lagrangian multipliers for stable neohooken
        self.rest_volume = ti.field(float, NT)  # rest volume of each tet
        self.inv_V = ti.field(float, NT)  # inverse volume of each tet
        self.alpha_tilde = ti.field(float, NT)
        self.alpha = ti.field(float, NT)

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
        self.is_fixed = ti.field(int, self.NV)
        self.fixed_pos = ti.Vector.field(3, float, self.NV)

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
        init_physics_kernel(
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
        init_alpha_kernel(self.rest_volume, self.args.mu, self.alpha)
        self.alpha_tilde_np = self.alpha_tilde.to_numpy()\
        
        self.reinit()



    def reinit(self):
        # args.reinit = "squash"
        # FIXME: no reinit will cause bug, why? FIXED: because when there is no deformation, the gradient will be in any direction! Sigma=(1,1,1) There will be singularity issue! We need to jump the constraint=0 case.
        # reinit pos
        self.initial_pos = self.pos.to_numpy()
        self.fixed_pos.copy_from(self.pos)
        self.fixed_stiffness = self.args.fixed_stiffness
        if args.reinit == "random":
            random_val = np.random.rand(self.pos.shape[0], 3)
            self.pos.from_numpy(random_val)
        elif args.reinit == "enlarge":
            self.pos.from_numpy(self.model_pos * 1.5)
        elif args.reinit == "squash":
            p = self.model_pos
            pymin = np.min(p[:, 1])
            p[:, 1] = pymin
            self.pos.from_numpy(p)
        elif args.reinit == "freefall":
            args.gravity = [0, -9.8, 0]
            self.gravity = ti.Vector(args.gravity)

            args.use_ground_collision = 1
 
            #for ground collision response
            min_pos_y = np.min(self.pos.to_numpy()[:,1])
            max_pos_y = np.max(self.pos.to_numpy()[:,1])
            if min_pos_y < 0.0:
                # move the model above the ground
                logging.warning(f"move the model above the ground")
                self.pos.from_numpy(self.pos.to_numpy() + np.array([0, -min_pos_y+(max_pos_y-min_pos_y)*0.01, 0]))
            self.ground_pos = 0.0
        elif args.reinit=="beam":
            p = self.model_pos
            # fix the beam at the left end
            # find the left end postion
            xmin = np.min(p, axis=0)[0]
            xmax = np.max(p, axis=0)[0]
            xsize = xmax - xmin 
            # a small region within the left end
            endregion = (xmin - xsize*0.01, xmin + xsize*0.01)
            # find the end particles where x is within the region
            self.fixed_particles = np.where((p[:, 0] > endregion[0]) & (p[:, 0] < endregion[1]))[0]

            # # # # set those particles inv_mass to 0
            # self.inv_mass_np = self.inv_mass.to_numpy()
            # self.inv_mass_np[self.fixed_particles] = 0.0
            # self.inv_mass.from_numpy(self.inv_mass_np)

            args.gravity = [0, -9.8, 0]
            self.gravity = ti.Vector(args.gravity)
            self.fixed_particles = np.array([873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 907, 911, 912, 914, 916, 918, 919, 922, 924, 928, 930, 932, 933, 935, 938, 940, 942, 943, 946, 947, 949, 950, 951, 952, 954, 955, 958, 960, 962, 963, 964, 965, 1993, 1996, 1997, 2000, 2001, 2004, 2005, 2009, 2010, 2012, 2015, 2017, 2019, 2021, 2022, 2024, 2028, 2029, 2030, 2033, 2036, 2038, 2039, 2040, 2043, 2044, 2045, 2050, 2051, 2052, 2053, 2054, 2057, 2062, 2063, 2064, 2067, 2068, 2069, 2070, 3797, 3813, 3815, 3817, 3819, 3821, 3826, 3830, 3833, 3835, 3837, 3838, 3842, 3846, 3847, 3853, 3855, 3857, 3860, 3862, 3866, 3901, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3929, 4011, 4016, 4018, 4019, 4020, 4021, 4024, 4027, 4029, 4032, 4033, 4034, 4035, 4040, 4041, 4042, 4044, 4045, 4048, 4049, 4052, 4056, 4058, 4060, 4062, 4064, 4065, 4066, 4067, 4071, 4074, 4075, 4076, 4077, 4079, 4080, 4081, 4082, 4083, 4084, 4087, 4089, 4090, 4093, 4094, 4095, 5126, 5164, 5167, 5170, 5178, 5186, 5211, 5216, 5224, 5251, 5258, 5259, 5265, 5275, 5285, 5286, 5288, 5290, 5293, 5294, 5296, 5297, 5298, 5299, 5301, 5302, 5303, 5306, 5308, 5311, 5323, 5330, 5331, 5425, 5431, 5434, 5435, 5437, 5438, 5439, 5441, 5442, 5444, 5445, 5446, 5447, 5449, 5453, 5473, 5475, 5476, 5480, 5602, 5651, 5669, 5746, 5754, 5793, 5856, 5876, 5889, 5903, 5939, 5987, 5999, 6090, 6092, 6180, 6346, 6367, 6453, 6735, 6788, 6807, 6847, 6861, 6912, 6951, 6971, 6997, 7124, 7155, 7156, 7266, 7300, 7311, 7375, 7402, 7539, 7647, 7655, 7694, 7712, 7802, 7837, 7879, 7925, 7931, 7941, 7962, 8106, 8113, 8143, 8165, 8170, 8172, 8191, 8226, 8284, 8291, 8321, 8322],dtype=np.int32) # set from args

            self.fixed_pos.from_numpy(self.pos.to_numpy())
            set_is_fixed_kernel(self.fixed_particles, self.is_fixed)
            # print(f"fixed particles: {self.fixed_particles}")
        
        elif args.reinit=="twist_bar":
            self.gravity = ti.Vector(args.gravity)
            deformed_pos = load_pos_from_node(args.load_file)
            self.pos.from_numpy(deformed_pos)



    # calc_dual use the base class's

    def calc_primal(self):
        def calc_primal_imply(self,G,M_inv):
            MASS = scipy.sparse.diags(1.0/(M_inv.diagonal()), format="csr")
            primary_residual = MASS @ (self.pos.to_numpy().flatten() - self.predict_pos.to_numpy().flatten()) - G.transpose() @ self.lagrangian.to_numpy()
            where_zeros = np.where(M_inv.diagonal()==0)
            primary_residual = np.delete(primary_residual, where_zeros)
            return primary_residual
        G = fill_G()
        primary_residual = calc_primal_imply(G, self.M_inv)
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
        if args.use_line_search:
            self.omega = self.line_search(self.pos.to_numpy(), self.dpos.to_numpy())
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


    def solveSoft_cuda_init(self):
        if hasattr(self, "softC"):
            return self.softC
        import pymgpbd as mp # type: ignore

        softC = mp.SolveSoft()

        softC.resize_fields(self.NV, self.NCONS)
        softC.pos = self.pos.to_numpy()
        softC.alpha_tilde = self.alpha_tilde.to_numpy()
        softC.vert = self.tet_indices.to_numpy()
        softC.inv_mass = self.inv_mass.to_numpy()
        softC.B = self.B.to_numpy()
        softC.lam = self.lagrangian.to_numpy()
        softC.delta_t = args.delta_t

        self.softC = softC

        from engine.util import vec_is_equal
        vec_is_equal(np.array(softC.B), self.B.to_numpy())

        return softC

    @timeit
    def solveSoft_cuda(self):
        softC = self.solveSoft_cuda_init()
        softC.pos = self.pos.to_numpy()

        softC.solve()
        
        # TODO: following data transfer will be removed
        c_ = softC.constraints
        gradC_ = softC.gradC
        b_ = softC.b

        c_1 = np.array(c_)
        gradC_1 = np.array(gradC_)
        b_1 = np.array(b_)
        
        self.constraints.from_numpy(c_1)
        self.gradC.from_numpy(gradC_1)
        self.b = b_1

        dlam, self.r_iter.r_Axb = self.linsol.run(self.b)
        self.dlam2dpos(dlam)


    # @timeit
    def solveSoft_python(self):
        self.pos_mid.copy_from(self.pos)
        self.compute_C_and_gradC()
        self.b = self.compute_b()
        dlam, self.r_iter.r_Axb = self.linsol.run(self.b)
        self.dlam2dpos(dlam)

    def solveSoft(self):
        self.solveSoft_python()

    def do_pre_iter0(self):
        self.update_constraints() # for calculation of r0
        self.dual0 = self.r_iter.calc_r0()
        return self.dual0

    @timeit
    def read_external_pos(self):
        if args.use_pintoanimation:
            self.read_geo_pinpos()
        if args.use_extra_spring or args.use_pintotarget:
            self.read_target_pos()

    def has_no_time_budget(self):
        self.frame_past_time = perf_counter() - self.tic_frame
        logging.info(f"FramePastTime: {self.frame_past_time*1000:.0f}ms")
        if args.solver_type=="AMG":
            if self.should_setup(): 
                self.frame_past_time = 0.0
        if self.frame_past_time > args.time_budget:
            logging.info(f"Time budget exceeded, break: frame past time: {self.frame_past_time:.2f}s, iter:{self.ite}")
            return True
        return False
    
    
    def AMG_calc_r(self):
        from engine.ti_kernels import calc_dual_kernel
        d  = calc_dual_kernel(self.alpha_tilde, self.lagrangian, self.constraints, self.dual_residual)
        return d
    

    # @timeit
    def do_external_constraints(self):
        if args.use_extra_spring:
            if self.ite ==0:
                self.extra_springs.aos.lam.fill(0.0)
            self.extra_springs.solve_one_iter(self.pos, self.target_pos, args.delta_t)
        if args.use_pintotarget:
            if self.ite ==0:
                self.pintotarget.solve(self.pos, self.target_pos, args.maxiter)
        if args.use_muscle2muscle:
            if self.ite ==0:
                self.m2mCons.aos.lam.fill(0.0)
            self.m2mCons.solve_one_iter(self.pos, args.delta_t)


    def  do_local_steps(self):
        for i in range(0, args.local_interval):
            project_constraints(
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
            self.dualr = np.linalg.norm(self.residual.to_numpy())
            print(f"{self.frame}-{self.ite}-local{i} loacl-step dual:{self.dualr:.2e}")

    
    def rayleigh_damping(self):
        for i in range(0, args.damping_steps):
            project_constraints(
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
            

    def substep_all_solver(self):
        self.tic_frame = time.perf_counter()
        semi_euler_kernel(args.delta_t, self.pos, self.predict_pos, self.old_pos, self.vel, args.damping_coeff, self.gravity)
        self.lagrangian.fill(0)
        self.log_energy(self.frame,0,f"{args.out_dir}/r/energy.txt")
        if args.use_external_constraints:
            self.read_external_pos()
            self.do_external_constraints()
        for self.ite in range(args.maxiter):
            # if self.has_no_time_budget():
            #     break
            self.tic_iter = perf_counter()
            self.solveSoft()
            self.log_energy(self.frame,self.ite+1,f"{args.out_dir}/r/energy.txt")
            self.toc_iter = perf_counter()
            
        self.collision_response()
        self.n_outer_all.append(self.ite+1)
        self.update_vel()


    def log_energy(self,frame, iter, filename_to_save=""):
        if args.calc_energy:
            te = compute_energy(self.inv_mass, self.pos, self.predict_pos, self.tet_indices, self.B, self.alpha, self.delta_t, self.is_fixed, self.fixed_stiffness, self.fixed_pos)
            s=f"Frame:{frame} Iter:{iter} Energy:{te:.8e}"
            print(s)
            with open(f"energy_twist_bar118k_{3*self.args.mu:.0e}_amg.txt", "a") as f:
                f.write(s+"\n")
            return te


    def do_post_iter(self):
        if self.args.export_matrix:
            export_all_levels_A(self)

        
    def substep_xpbd(self):
        semi_euler_kernel(args.delta_t, self.pos, self.predict_pos, self.old_pos, self.vel, args.damping_coeff, self.gravity)
        # self.semi_euler()
        self.lagrangian.fill(0)
        self.lagrangian_D.fill(0)
        self.log_energy(self.frame,0,f"{args.out_dir}/r/energy.txt")
        if args.use_external_constraints:
            self.do_external_constraints()
        for self.ite in range(args.maxiter):
            if  args.constuitive_model == "StableNeoHookean":
                project_constraints_stableNeohookean_kernel(
                    self.tet_indices,
                    self.B,
                    self.inv_mass,
                    self.lagrangian,
                    self.lagrangian_D,
                    self.pos,
                    args.omega,
                    args.mu,
                    args.lame_lambda,
                    args.delta_t,
                    self.rest_volume
                )
            else:
                project_constraints_arap_v2_kernel(
                    self.pos_mid,
                    self.tet_indices,
                    self.inv_mass,
                    self.lagrangian,
                    self.B,
                    self.pos,
                    self.alpha_tilde,
                    self.residual,
                    args.omega
                )
            self.log_energy(self.frame,self.ite+1,f"{args.out_dir}/r/energy.txt")

        self.collision_response()
        self.n_outer_all.append(self.ite+1)
        update_vel(args.delta_t, self.pos, self.old_pos, self.vel)



@ti.kernel
def set_is_fixed_kernel(fixed_particles: ti.types.ndarray(dtype=ti.i32), is_fixed: ti.template()):
    for i in range(fixed_particles.shape[0]):
        is_fixed[fixed_particles[i]] = 1


@ti.kernel
def update_vel(delta_t: ti.f32, pos: ti.template(), old_pos: ti.template(), vel: ti.template()):
    for i in pos:
        vel[i] = (pos[i] - old_pos[i]) / delta_t


@ti.kernel
def calc_dual_residual(alpha_tilde:ti.template(),
                       lagrangian:ti.template(),
                       constraint:ti.template(),
                       dual_residual:ti.template()):
    for i in range(dual_residual.shape[0]):
        dual_residual[i] = -(constraint[i] + alpha_tilde[i] * lagrangian[i])


@ti.kernel
def project_constraints(
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
    for i in pos:
        pos_mid[i] = pos[i]

    # ti.loop_config(serialize=meta.serialize)
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
            denorminator = (
                inv_mass[p0] * g0.norm_sqr()
                + inv_mass[p1] * g1.norm_sqr()
                + inv_mass[p2] * g2.norm_sqr()
                + inv_mass[p3] * g3.norm_sqr()
            )
            residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t])
            dlambda[t] = residual[t] / (denorminator + alpha_tilde[t])

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



# use global memory variables as less as possible
# %14 speed up compared to the previous version
# 0.5ms per iter for 12K ele bunny small dt=3ms mu=1e6
@ti.kernel
def project_constraints_arap_v2_kernel(
    pos_mid: ti.template(),
    tet_indices: ti.template(),
    inv_mass: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    pos: ti.template(),
    alpha_tilde: ti.template(),
    # constraint: ti.template(),
    residual: ti.template(),
    # gradC: ti.template(),
    # dlambda: ti.template(),
    # dpos: ti.template(),
    omega: ti.f32
):
    for i in pos:
        pos_mid[i] = pos[i]

    # ti.loop_config(serialize=meta.serialize)
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]

        x0, x1, x2, x3 = pos_mid[p0], pos_mid[p1], pos_mid[p2], pos_mid[p3]
        

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint1 = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        if constraint1 > 1e-6:
            g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
            # gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0, g1, g2, g3
            denorminator = (
                inv_mass[p0] * g0.norm_sqr()
                + inv_mass[p1] * g1.norm_sqr()
                + inv_mass[p2] * g2.norm_sqr()
                + inv_mass[p3] * g3.norm_sqr()
            )
            residual[t] = -(constraint1 + alpha_tilde[t] * lagrangian[t])
            dlambda1 =  -(constraint1 + alpha_tilde[t] * lagrangian[t]) / (denorminator + alpha_tilde[t])

            lagrangian[t] += dlambda1
            pos[p0] += omega * inv_mass[p0] * dlambda1 * g0
            pos[p1] += omega * inv_mass[p1] * dlambda1 * g1
            pos[p2] += omega * inv_mass[p2] * dlambda1 * g2
            pos[p3] += omega * inv_mass[p3] * dlambda1 * g3



@ti.kernel
def project_constraints_stableNeohookean_kernel(
        tet_indices: ti.template(),
        B: ti.template(),
        inv_mass: ti.template(),
        lagrangian_H: ti.template(),
        lagrangian_D: ti.template(),
        pos: ti.template(),
        omega: ti.f32,
        mu: ti.f32,
        lame_lambda: ti.f32,
        dt: ti.f32,
        rest_volume: ti.template(),
    ):
        gamma = 1 + mu/lame_lambda  # stable neo-hookean

        for i in tet_indices:
            alpha_tilde_H = 1.0/(dt*dt*lame_lambda*rest_volume[i])
            alpha_tilde_D = 1.0/(dt*dt*mu*rest_volume[i])


            ia, ib, ic, id = tet_indices[i]
            a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id]
            invM0, invM1, invM2, invM3 = inv_mass[ia], inv_mass[ib], inv_mass[ic], inv_mass[id]
            D_s = ti.Matrix.cols([b - a, c - a, d - a])
            F = D_s @ B[i]

            # Constraint 1
            C_H = F.determinant() - gamma
            f1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
            f2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
            f3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])

            f23 = f2.cross(f3)
            f31 = f3.cross(f1)
            f12 = f1.cross(f2)
            f = ti.Vector([f23[0], f23[1], f23[2], f31[0], f31[1], f31[2], f12[0], f12[1], f12[2]])
            dFdp1T = make_matrix(B[i][0, 0], B[i][0, 1], B[i][0, 2])
            dFdp2T = make_matrix(B[i][1, 0], B[i][1, 1], B[i][1, 2])
            dFdp3T = make_matrix(B[i][2, 0], B[i][2, 1], B[i][2, 2])

            g1 = dFdp1T @ f
            g2 = dFdp2T @ f
            g3 = dFdp3T @ f
            g0 = -g1 - g2 - g3
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr() + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
            dLambda = (-C_H - alpha_tilde_H * lagrangian_H[i]) / (l + alpha_tilde_H)
            lagrangian_H[i] += dLambda
            pos[ia] += omega * invM0 * dLambda * g0
            pos[ib] += omega * invM1 * dLambda * g1
            pos[ic] += omega * invM2 * dLambda * g2
            pos[id] += omega * invM3 * dLambda * g3

            # Constraint 2
            C_D = sqrt(f1.norm_sqr() + f2.norm_sqr() + f3.norm_sqr())
            if C_D < 1e-6:
                continue
            r_s = 1.0 / C_D
            f = ti.Vector([f1[0], f1[1], f1[2], f2[0], f2[1], f2[2], f3[0], f3[1], f3[2]])
            g1 = r_s * (dFdp1T @ f)
            g2 = r_s * (dFdp2T @ f)
            g3 = r_s * (dFdp3T @ f)
            g0 = r_s * (-g1 - g2 - g3)
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr() + invM2 * g2.norm_sqr() + invM3 * g3.norm_sqr()
            dLambda = (-C_D - alpha_tilde_D * lagrangian_D[i]) / (l + alpha_tilde_D)
            lagrangian_D[i] += dLambda
            pos[ia] += omega * invM0 * dLambda * g0
            pos[ib] += omega * invM1 * dLambda * g1
            pos[ic] += omega * invM2 * dLambda * g2
            pos[id] += omega * invM3 * dLambda * g3


@ti.kernel
def rayleigh_damping_kernel(
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
    omega: ti.f32,
    predict_pos: ti.template(),
    delta_t: ti.f32,
    beta_tilde: ti.template(),
):
    for i in pos:
        pos_mid[i] = pos[i]

    # ti.loop_config(serialize=meta.serialize)
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]

        x0, x1, x2, x3 = pos_mid[p0], pos_mid[p1], pos_mid[p2], pos_mid[p3]
        px0, px1, px2, px3 = predict_pos[p0], predict_pos[p1], predict_pos[p2], predict_pos[p3]
        px0diff, px1diff, px2diff, px3diff = px0 - x0, px1 - x1, px2 - x2, px3 - x3

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        constraint[t] = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        if constraint[t] > 1e-6:
            g0, g1, g2, g3 = compute_gradient(U, S, V, B[t])
            gradC[t, 0], gradC[t, 1], gradC[t, 2], gradC[t, 3] = g0, g1, g2, g3
            # denorminator = (
            #     inv_mass[p0] * g0.norm_sqr()
            #     + inv_mass[p1] * g1.norm_sqr()
            #     + inv_mass[p2] * g2.norm_sqr()
            #     + inv_mass[p3] * g3.norm_sqr()
            # )

            # from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=13009
            ldiv = 0.0
            damp = 0.0
            ldiv += inv_mass[p0] * ti.math.dot(g0, g0)
            ldiv += inv_mass[p1] * ti.math.dot(g1, g1)
            ldiv += inv_mass[p2] * ti.math.dot(g2, g2)
            ldiv += inv_mass[p3] * ti.math.dot(g3, g3)
            damp += ti.math.dot((px0diff), g0)
            damp += ti.math.dot((px1diff), g1)
            damp += ti.math.dot((px2diff), g2)
            damp += ti.math.dot((px3diff), g3)
            gamma = alpha_tilde[t] * beta_tilde[t] / delta_t
            ldiv = ldiv * (1.0 + gamma) + alpha_tilde[t]
            dlambda[t] = (-constraint[t]- alpha_tilde[i]*lagrangian[i] - gamma * damp ) / ldiv
            residual[t] = -(constraint[t] + alpha_tilde[t] * lagrangian[t]) 

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
def semi_euler_kernel(
    delta_t: ti.f32,
    pos: ti.template(),
    predict_pos: ti.template(),
    old_pos: ti.template(),
    vel: ti.template(),
    damping_coeff: ti.f32,
    gravity: ti.template(),
):
    for i in pos:
        vel[i] += delta_t * gravity
        vel[i] *= damping_coeff
        old_pos[i] = pos[i]
        pos[i] += delta_t * vel[i]
        predict_pos[i] = pos[i]

@ti.kernel
def reset_lagrangian(lagrangian: ti.template()):
    for i in lagrangian:
        lagrangian[i] = 0.0




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
def init_alpha_kernel(
    rest_volume: ti.template(),
    mu: ti.f32,
    alpha: ti.template(),
):
    for i in alpha:
        alpha[i] =  1.0/mu /rest_volume[i]


@ti.kernel
def init_physics_kernel(
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
    if args.mass_density > 0.0:
        print("Using mass density: ", args.mass_density)
        for i in tet_indices:
            ia, ib, ic, id = tet_indices[i]
            tet_mass = args.mass_density * rest_volume[i]
            avg_mass = tet_mass / 4.0
            mass[ia] += avg_mass
            mass[ib] += avg_mass
            mass[ic] += avg_mass
            mass[id] += avg_mass
        for i in inv_mass:
            inv_mass[i] = 1.0/mass[i]
    elif args.total_mass > 0.0:
        for i in tet_indices:
            ia, ib, ic, id = tet_indices[i]
            mass_density = args.total_mass / total_volume
            tet_mass = mass_density * rest_volume[i]
            avg_mass = tet_mass / 4.0
            mass[ia] += avg_mass
            mass[ib] += avg_mass
            mass[ic] += avg_mass
            mass[id] += avg_mass
        for i in range(inv_mass.shape[0]):
            inv_mass[i] = 1.0 / mass[i]
            inv_mass[i] = 1.0 / mass[i]
    elif args.pmass > 0.0:
        for i in inv_mass:
            inv_mass[i] = 1.0/args.pmass
    else:
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
def calc_dualnorm_kernel(
    pos_temp: ti.types.ndarray(dtype=ti.math.vec3),
    tet_indices: ti.template(),
    lagrangian: ti.template(),
    B: ti.template(),
    alpha_tilde: ti.template(),
) -> ti.f32:
    dual = 0.0
    for t in range(tet_indices.shape[0]):
        p0 = tet_indices[t][0]
        p1 = tet_indices[t][1]
        p2 = tet_indices[t][2]
        p3 = tet_indices[t][3]
        x0, x1, x2, x3 = pos_temp[p0], pos_temp[p1], pos_temp[p2], pos_temp[p3]

        D_s = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F = D_s @ B[t]
        U, S, V = ti.svd(F)
        c = ti.sqrt((S[0, 0] - 1) ** 2 + (S[1, 1] - 1) ** 2 + (S[2, 2] - 1) ** 2)
        r = -(c + alpha_tilde[t] * lagrangian[t])
        dual += r * r
    return ti.sqrt(dual)





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
    if lv==0:
        extlib.fastmg_fetch_A(lv, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr)
        A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(matsize, matsize))
    else:
        data = np.zeros(nnz, dtype=np.float32)
        indices = np.zeros(nnz, dtype=np.int32)
        indptr = np.zeros(matsize+1, dtype=np.int32)
        extlib.fastmg_fetch_A(lv, data, indices, indptr)
        A = scipy.sparse.csr_matrix((data, indices, indptr), shape=(matsize, matsize))
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


def export_all_levels_A(ist):
    from engine.util import export_A_b
    AMG_A()
    nl = ist.linsol.get_nl()
    for l in range(nl):
        print(f"exporting A of level {l}...")
        A = fetch_A_from_cuda(l)
        print(f"A.shape={A.shape}")
        export_A_b(A, None, dir=args.out_dir+"/A/", postfix=f"L{l}")
        print(f"exported A of level {l}...")
    print("exported all levels A and exit..")
    exit(0)

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
                should_setup=ist.should_setup,
                fill_A_in_cuda=AMG_A,
                graph_coloring=gc,
                copy_A=True,
            )
        else:
            linsol = AmgPython(args, get_A0_python, ist.should_setup)
    elif args.solver_type == "AMGX":
        linsol = AmgxSolver(args.amgx_config, get_A0_python, args.cuda_dir, args.amgx_lib_dir)
    elif args.solver_type == "DIRECT":
        if args.direct_solver_type=="pardiso":
            from engine.solver.direct_solver import DirectSolverPardiso
            linsol = DirectSolverPardiso(get_A0_cuda)
        elif args.use_cuda:
            linsol = AmgCuda(
                    args=args,
                    extlib=extlib,
                    get_A0=get_A0_cuda,
                    should_setup=ist.should_setup,
                    fill_A_in_cuda=AMG_A,
                    only_direct=True,
                    copy_A=True,
                )
        else:
            linsol = DirectSolver(get_A0_python)
    elif args.solver_type == "XPBD":
        linsol=None
    else:
        linsol=None
    return linsol


def get_rbm(pos): 
    coo = pos.flatten().astype(np.float64)
    rbm = np.zeros(coo.shape[0]*6, dtype=np.float64)
    extlib.fastmg_calc_rbm(coo, coo.size, rbm)
    rbm = rbm.reshape(-1,6)
    return rbm






def load_pos_from_txt(filename):
    pos = np.loadtxt(filename,dtype=np.float32).reshape(-1, 3)
    return pos

def  load_pos_from_node(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)
    return pos



def init():
    tic = perf_counter()
    global args
    args = init_args()
    process_dirs(args)
    init_logger(args)
    global extlib
    extlib = init_extlib(args,sim="soft")
    global ist
    ist = SoftBody(args.model_path)
    ist.linsol = init_linear_solver()
    if args.solver_type != "XPBD" and args.solver_type != "NEWTON":
        from engine.soft.fill_A import init_direct_fill_A
        init_direct_fill_A(ist,extlib)
    print(f"initialize time:", perf_counter()-tic)


def main():
    init()
    from engine.util import main_loop
    main_loop(ist,args)

if __name__ == "__main__":
    main()
