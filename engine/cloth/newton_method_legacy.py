import numpy as np
import scipy.io
import scipy.sparse
import time
import sys, os
import taichi as ti
import taichi.math as tm
import logging

sys.path.append(os.getcwd())

from engine.solver.amg_cuda import AmgCuda
from engine.cloth.cloth3d import Cloth
from engine.cloth.constraints import SetupConstraints, ConstraintType
from engine.util import timeit, norm, norm_sqr, normalize, debug, debugmat,csr_is_equal
from engine.physical_base import PhysicalBase
from engine.line_search import LineSearch

@ti.data_oriented
class NewtonMethod(Cloth):
    def __init__(self,args,extlib):
        super().__init__(args)
        self.EPSILON = 1e-9
        self.set_mass()
        self.stiffness = self.set_stiffness(self.alpha_tilde, self.delta_t)
        self.vert =self.edge

        # self.linsol = AmgCuda(
        #         args=args,
        #         extlib=extlib,
        #         get_A0= lambda: self.hessian,
        #         copy_A=False,
        #         should_setup=self.should_setup,
        #     )
        self.linsol = AmgCuda(
            args=args,
            extlib=extlib,
            get_A0= lambda: self.hessian,
            only_jacobi=True,
        )
        
        self.use_line_search=True
        self.ls_alpha=0.25
        self.ls_beta=0.1
        self.ls_step_size=1.0

    def set_stiffness(self, alpha_tilde, delta_t):
        stiffness = ti.field(dtype=ti.f32, shape=alpha_tilde.shape[0])
        @ti.kernel
        def kernel(stiffness:ti.template(),
                   alpha_tilde:ti.template()):
            for i in alpha_tilde:
                stiffness[i] = 1.0/(alpha_tilde[i]*delta_t**2)
        kernel(stiffness, alpha_tilde)
        return stiffness
    
    def set_mass(self):
        from engine.util import set_mass_matrix_from_invmass
        self.MASS = set_mass_matrix_from_invmass(self.inv_mass)
        ...

    @timeit
    def evaluateGradient(self, x):
        return self.calc_gradient_cloth_imply_ti(x)  

    @timeit
    def evaluateHessian(self, x):
        return self.calc_hessian_imply_ti(x)

    @timeit
    def substep_newton(self):
        self.semi_euler()
        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter(self.pos)
            if converge:
                break
        self.n_outer_all.append(self.ite+1)
        self.update_vel()

    # https://github.com/chunleili/fast_mass_spring/blob/a203b39ae8f5ec295c242789fe8488dfb7c42951/fast_mass_spring/source/simulation.cpp#L510
    # integrateNewtonDescentOneIteration
    def step_one_iter(self, x):
        print(f'ite: {self.ite}')   
        gradient = self.evaluateGradient(x)
        nrmsqr = norm_sqr(gradient)
        if nrmsqr < self.EPSILON:
            print(f'gradient nrmsqr {nrmsqr} <EPSILON')
            return True

        self.hessian = self.evaluateHessian(x)

        descent_dir,r_Axb = self.linsol.run(gradient.flatten())
        descent_dir = -descent_dir.reshape(-1,3)
        logging.info(f"    r_Axb: {r_Axb[0]:.2e} {r_Axb[-1]:.2e}")

        step_size = self.line_search(x, self.predict_pos, gradient, descent_dir)
        logging.info (f"    step_size: {step_size:.2e}")

        # x += descent_dir.reshape(-1,3) * step_size
        x.from_numpy(x.to_numpy() + descent_dir.reshape(-1,3) * step_size)

        if step_size < self.EPSILON:
            print(f'step_size {step_size} <EPSILON')
            return True
        else:
            return False


    def calc_energy(self,x, predict_pos):

        def update_constraints(x):
            @ti.kernel
            def update_constraints_kernel(
                pos:ti.template(),
                edge:ti.template(),
                rest_len:ti.template(),
                constraints:ti.template(),
            ):
                for i in range(edge.shape[0]):
                    idx0, idx1 = edge[i]
                    dis = pos[idx0] - pos[idx1]
                    constraints[i] = dis.norm() - rest_len[i]
            update_constraints_kernel(x, self.edge, self.rest_len, self.constraints)


        def compute_inertial_energy(x, predict_pos)->float:
            @ti.kernel
            def compute_inertial_energy_kernel(
                pos: ti.template(),
                predict_pos: ti.template(),
                inv_mass: ti.template(),
                delta_t: ti.f32,
            )->ti.f32:
                inertial_energy = 0.0
                inv_h2 = 1.0 / delta_t**2
                for i in range(pos.shape[0]):
                    if inv_mass[i] == 0.0:
                        continue
                    inertial_energy += 0.5 / inv_mass[i] * (pos[i] - predict_pos[i]).norm_sqr() * inv_h2
                return inertial_energy
        
            res = compute_inertial_energy_kernel(x, predict_pos, self.inv_mass, self.delta_t)
            return res

        update_constraints(x)
        self.potential_energy = self.compute_potential_energy()
        self.inertial_energy = compute_inertial_energy(x, predict_pos)
        self.energy = self.potential_energy + self.inertial_energy
        logging.info(f"    energy: {self.energy:.8e}")
        return self.energy


    

    def calc_gradient_cloth_imply_ti(self, x):
        # assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        NV = x.shape[0]
        x_tilde = self.predict_pos
        inv_mass = self.inv_mass
        delta_t = self.delta_t
        
        gradient = np.zeros((NV, 3), dtype=np.float32)

        @ti.kernel
        def kernel(x:ti.template(),
                   vert:ti.template(),
                   rest_len:ti.template(),
                   NCONS:ti.i32,
                   gradient:ti.types.ndarray(dtype=tm.vec3),
                   stiffness:ti.template(),
                   ):
            for i in range(NCONS):
                i0, i1 = vert[i]
                x_ij = x[i0] - x[i1]
                l_ij = x_ij.norm()
                g_ij = stiffness[i] * (l_ij - rest_len[i]) * x_ij.normalized()
                if inv_mass[i0] != 0.0:
                    gradient[i0] += g_ij
                if inv_mass[i1] != 0.0:
                    gradient[i1] -= g_ij

        @ti.kernel
        def kernel2(x:ti.template(),
                   x_tilde:ti.template(),
                   inv_mass: ti.template(),
                   delta_t:ti.f32,
                   gradient:ti.types.ndarray(dtype=tm.vec3),
                   ):
            for i in range(inv_mass.shape[0]):
                if inv_mass[i] !=0.0:
                    # gradient[i] = delta_t*delta_t
                    gradient[i] += (1.0/self.inv_mass[i] * (x[i]-x_tilde[i]) )/delta_t**2
                
        kernel(x,vert,rest_len, NCONS, gradient, stiffness)

        g2 = gradient.copy()
        kernel2(x,x_tilde,inv_mass,delta_t,g2)
        
        return g2
    

    def calc_hessian_imply_ti(self, x) -> scipy.sparse.csr_matrix:
        # assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        NV = x.shape[0]
        delta_t = self.delta_t
        MASS = self.MASS
        
        MAX_NNZ = NCONS* 50     # estimate the nnz: 3*3*4*NCONS

        ii = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        jj = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        vv = np.zeros(dtype=np.float32,shape=MAX_NNZ)


        @ti.kernel
        def kernel(x:ti.template(),
                   vert:ti.template(),
                   rest_len:ti.template(),
                   stiffness:ti.template(),
                   NCONS:ti.i32,
                   ii:ti.types.ndarray(),
                   jj:ti.types.ndarray(),
                   vv:ti.types.ndarray(),
                   ):
            kk = 0
            for i in range(NCONS):
                p1, p2 = vert[i]
                x_ij = x[p1] - x[p2]
                # l_ij = norm(x_ij)
                l_ij = x_ij.norm()
                l0 = rest_len[i]
                ks = stiffness[i]
                k = ks * (tm.eye(3) - l0/l_ij*(tm.eye(3) - x_ij.outer_product(x_ij)/(l_ij*l_ij)))
                # k = ks * (np.eye(3) - l0/l_ij*(np.eye(3) - np.outer(x_ij, x_ij)/(l_ij*l_ij)))
                for row in ti.static(range(3)):
                    for col in ti.static(range(3)):
                        val = k[row, col]
                        ii[kk] = 3*p1 + row
                        jj[kk] = 3*p1 + col
                        vv[kk] = val
                        kk += 1
                        ii[kk] = 3*p1 + row
                        jj[kk] = 3*p2 + col
                        vv[kk] = -val
                        kk += 1
                        ii[kk] = 3*p2 + row
                        jj[kk] = 3*p1 + col
                        vv[kk] = -val
                        kk += 1
                        ii[kk] = 3*p2 + row
                        jj[kk] = 3*p2 + col
                        vv[kk] = val
                        kk += 1


        kernel(x,vert,rest_len, stiffness, NCONS, ii, jj, vv)
        hessian = scipy.sparse.coo_matrix((vv,(ii,jj)),shape=(NV*3, NV*3),dtype=np.float32)
        hessian =  MASS/(delta_t * delta_t) +hessian
        hessian = hessian.tocsr()
        return hessian


    def line_search(self, x, predict_pos, gradient_dir, descent_dir):
        if not self.use_line_search:
            return self.ls_step_size
        
        x_plus_tdx = self.copy_field(x)
        descent_dir = descent_dir.reshape(-1,3)

        t = 1.0/self.ls_beta
        currentObjectiveValue = self.calc_energy(x, predict_pos)
        ls_times = 0
        while ls_times==0 or (lhs >= rhs and t > self.EPSILON):
            t *= self.ls_beta
            # x_plus_tdx = (x.flatten() + t*descent_dir).reshape(-1,3)
            self.calc_x_plus_tdx(x_plus_tdx, x, t, descent_dir)
            lhs = self.calc_energy(x_plus_tdx, predict_pos)
            rhs = currentObjectiveValue + self.ls_alpha * t * np.dot(gradient_dir.flatten(), descent_dir.flatten())
            ls_times += 1
        self.energy = lhs
        print(f'    energy: {self.energy:.8e}')
        print(f'    ls_times: {ls_times}')

        if t < self.EPSILON:
            t = 0.0
        else:
            self.ls_step_size = t
        return t
    
    @staticmethod
    def calc_x_plus_tdx(x_plus_tdx, x, t, descent_dir):
        @ti.kernel
        def kernel( x_plus_tdx:ti.template(),
                    x:ti.template(),
                    t:ti.f32,
                    descent_dir:ti.types.ndarray(dtype=tm.vec3)):
            for i in range(x_plus_tdx.shape[0]):
                x_plus_tdx[i] = x[i] + t * descent_dir[i]
        kernel(x_plus_tdx, x, t, descent_dir)
    

    @staticmethod
    def copy_field(f1):
        f2 = ti.Matrix.field(n=f1.n, m=f1.m,ndim=f1.ndim, dtype=f1.dtype, shape=f1.shape)
        @ti.kernel
        def copy_kernel(f1:ti.template(), f2:ti.template()):
            for i in f1:
                f2[i] = f1[i]
        copy_kernel(f1, f2)
        return f2

