import numpy as np
import scipy.io
import scipy.sparse
import time
import sys, os
import taichi as ti
import taichi.math as tm
import logging

sys.path.append(os.getcwd())

from engine.solver.direct_solver import DirectSolver
from engine.cloth.cloth3d import Cloth
from engine.cloth.constraints import SetupConstraints, ConstraintType
from engine.util import timeit, norm, norm_sqr, normalize, debug, debugmat,csr_is_equal
from engine.physical_base import PhysicalBase
from engine.line_search import LineSearch

@ti.data_oriented
class NewtonMethod(Cloth):
    def __init__(self,args):
        super().__init__(args)

        self.EPSILON = 1e-15

        self.inertial_y = self.predict_pos.to_numpy()

        # self.setupConstraints = SetupConstraints(self.pos.to_numpy(), self.edge.to_numpy())
        # self.constraintsNew = self.setupConstraints.constraints
        # self.adapter = self.setupConstraints.adapter #from AOS to SOA

        # self.set_mass()
        self.stiffness = self.set_stiffness(self.alpha_tilde, self.delta_t)
        self.vert =self.edge

        def get_A():
            return self.hessian
        self.linear_solver = DirectSolver(get_A)


        # self.calc_hessian_imply_ti = CalculateHessianTaichi(self.stiffness, self.rest_len, self.edge, self.MASS, self.delta_t).run
        # self.calc_hessian_imply_py = CalculateHessianPython(self.constraintsNew, self.MASS, self.delta_t).run

        # self.calc_gradient_imply_ti = CalculateGradientTaichi(self.stiffness, self.rest_len, self.edge, self.MASS, self.delta_t, self.inertial_y).run
        # self.calc_gradient_imply_py = CalculateGradientPython(self.constraintsNew, self.MASS, self.delta_t, self.inertial_y).run

        self.calc_obj_func_imply_ti = self.calc_energy
        # self.calc_obj_func_imply_py = CalculateObjectiveFunctionPython(self.constraintsNew, self.MASS, self.delta_t, self.inertial_y).run

        ls = LineSearch(self.calc_obj_func_imply_ti)
        self.line_search = ls.line_search

    def set_stiffness(self, alpha_tilde, delta_t):
        stiffness = ti.field(dtype=ti.f32, shape=alpha_tilde.shape[0])
        @ti.kernel
        def kernel(stiffness:ti.template(),
                   alpha_tilde:ti.template()):
            for i in alpha_tilde:
                stiffness[i] = 1.0/(alpha_tilde[i]*delta_t**2)
        kernel(stiffness, alpha_tilde)
        return stiffness
    
    # def set_mass(self):
        # # pmass = 1.0 / self.NV
        # pmass = 1.0
        # self.MASS = scipy.sparse.diags([pmass]*self.NV*3)
        # self.inv_mass.fill(1.0)

    @timeit
    def evaluateGradient(self, x):
        return self.calc_gradient_imply_ti(x)  

    @timeit
    def evaluateHessian(self, x):
        return self.calc_hessian_imply_ti(x)

    @timeit
    def substep_newton(self):
        self.semi_euler()
        self.inertial_y = self.predict_pos.to_numpy()
        pos_next = self.pos.to_numpy().copy()
        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter(pos_next)
            if converge:
                break
        self.n_outer_all.append(self.ite+1)
        self.pos.from_numpy(pos_next)
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

        self.eliminate_zero_inv_mass(gradient,self.hessian, self.inv_mass)

        descent_dir,_ = self.linear_solver.run(gradient)
        descent_dir = -descent_dir

        step_size = self.line_search(x, self.inertial_y , gradient, descent_dir)

        x += descent_dir.reshape(-1,3) * step_size

        if step_size < self.EPSILON:
            print(f'step_size {step_size} <EPSILON')
            return True
        else:
            return False




    def update_constraints(self,x):
        update_constraints_kernel(x, self.edge, self.rest_len, self.constraints)

    def calc_energy(self,x, predict_pos):
        x = x.astype(np.float32)
        predict_pos = predict_pos.astype(np.float32)
        self.update_constraints(x)
        self.potential_energy = self.compute_potential_energy()
        self.inertial_energy = self.compute_inertial_energy(x, predict_pos)
        self.energy = self.potential_energy + self.inertial_energy
        # print(f"potential_energy: {self.potential_energy}")
        # print(f"inertial_energy: {self.inertial_energy}")
        logging.info(f"    energy: {self.energy:.8e}")
        return self.energy

    def compute_inertial_energy(self,x, predict_pos)->float:
        res = compute_inertial_energy_kernel(x, predict_pos, self.inv_mass, self.delta_t)
        return res
    

    def calc_gradient_imply_ti(self, x):
        assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        NV = x.shape[0]
        x_tilde = self.inertial_y
        inv_mass = self.inv_mass
        delta_t = self.delta_t
        
        gradient = np.zeros((NV, 3), dtype=np.float32)

        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
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
                gradient[i0] += g_ij
                gradient[i1] -= g_ij

        @ti.kernel
        def kernel2(x:ti.types.ndarray(dtype=tm.vec3),
                   x_tilde:ti.types.ndarray(dtype = tm.vec3),
                   inv_mass: ti.template(),
                   delta_t:ti.f32,
                   gradient:ti.types.ndarray(dtype=tm.vec3),
                   ):
            for i in range(inv_mass.shape[0]):
                if inv_mass[i] !=0.0:
                    gradient[i] *= delta_t*delta_t
                    gradient[i] += 1.0/self.inv_mass[i] * (x[i]-x_tilde[i])
                
        kernel(x,vert,rest_len, NCONS, gradient, stiffness)

        g2 = gradient.copy()
        kernel2(x,x_tilde,inv_mass,delta_t,g2)
        
        return g2.flatten()
    

    def calc_hessian_imply_ti(self, x) -> scipy.sparse.csr_matrix:
        assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        NV = x.shape[0]
        delta_t = self.delta_t
        inv_mass = self.inv_mass
        
        MAX_NNZ = NCONS* 50     # estimate the nnz: 3*3*4*NCONS

        ii = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        jj = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        vv = np.zeros(dtype=np.float32,shape=MAX_NNZ)


        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
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

        def kernel2(hdiag:ti.types.ndarray(),
                    inv_mass:ti.template()):
            for i in range(hdiag.shape[0]):
                if inv_mass[i//3]!=0:
                    hdiag[i] += 1.0/inv_mass[i//3]

        
        kernel(x,vert,rest_len, stiffness, NCONS, ii, jj, vv)
        hessian = scipy.sparse.coo_matrix((vv,(ii,jj)),shape=(NV*3, NV*3),dtype=np.float32)
        hessian = delta_t * delta_t * hessian

        # H += MASS
        h2 = hessian.copy()
        h2diag = h2.diagonal()
        kernel2(h2diag,inv_mass)
        h2.setdiag(h2diag)

        return h2.tocsr


@ti.kernel
def update_constraints_kernel(
    pos:ti.types.ndarray(dtype=tm.vec3),
    edge:ti.template(),
    rest_len:ti.template(),
    constraints:ti.template(),
):
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]

@ti.kernel
def compute_inertial_energy_kernel(
    pos: ti.types.ndarray(dtype=tm.vec3),
    predict_pos: ti.types.ndarray(dtype=tm.vec3),
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







class CalculateObjectiveFunctionTaichi():
    def __init__(self, vert, rest_len, stiffness, NCONS, MASS, delta_t, inertial_y):
        self.vert = vert
        self.rest_len = rest_len
        self.stiffness = stiffness
        self.NCONS = NCONS
        self.MASS = MASS
        self.delta_t = delta_t
        self.inertial_y = inertial_y

    def run(self, x):
        return self.calc_obj_func_imply_ti(x)

    def calc_obj_func_imply_ti(self,x) -> float:
        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
                   vert:ti.template(),
                   rest_len:ti.template(),
                   NCONS:ti.i32,
                   stiffness:ti.template(),
                   )->ti.f32:
            potential = 0.0
            for i in range(NCONS):
                x_ij = x[vert[i][0]] - x[vert[i][1]]
                l_ij = ti.cast((x_ij).norm(), ti.f32)
                l0 = rest_len[i]
                
                potential += 0.5 * stiffness[i] * (l_ij - l0) ** 2
            return potential
        
        potential_term = kernel(x,self.vert,self.rest_len, self.NCONS, self.stiffness)

        x_diff = x.flatten() - self.inertial_y.flatten()
        inertia_term = 0.5 * x_diff.transpose() @ self.MASS @ x_diff

        h_square = self.delta_t * self.delta_t
        # res = inertia_term + potential_term * h_square #fast mass spring
        res = inertia_term/h_square + potential_term
        return res     


class CalculateObjectiveFunctionPython():
    def __init__(self, constraintsNew, MASS, delta_t, inertial_y):
        self.constraintsNew = constraintsNew
        self.MASS = MASS
        self.delta_t = delta_t
        self.inertial_y = inertial_y

    def run(self, x):
        return self.calc_obj_func_imply_py(x)

    def calc_obj_func_imply_py(self, x):
        potential_term = 0.0
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                potential_term += self.EvaluatePotentialEnergyAttachment(c, x.reshape(-1,3))
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                potential_term += self.EvaluatePotentialEnergyDistance(c, x.reshape(-1,3))

        x_diff = x.flatten() - self.inertial_y.flatten()
        inertia_term = 0.5 * x_diff.transpose() @ self.MASS @ x_diff

        h_square = self.delta_t * self.delta_t
        # res = inertia_term + potential_term * h_square #fast mass spring
        res = inertia_term/h_square + potential_term
        return res

    # // 0.5*k*(current_length)^2
    def EvaluatePotentialEnergyAttachment(self, c, x):
        assert x.shape[1] == 3
        res = 0.5 * c.stiffness * norm_sqr(x[c.p0] - c.fixed_point)
        return res

    # // 0.5*k*(current_length - rest_length)^2
    def EvaluatePotentialEnergyDistance(self, c, x):
        assert x.shape[1] == 3
        x_ij = x[c.p1] - x[c.p2]
        l_ij = norm(x_ij)
        l0 = c.rest_len
        res = 0.5 * c.stiffness * (l_ij - l0) ** 2
        return res 


class CalculateHessianTaichi():
    def __init__(self, stiffness, rest_len, vert, inv_mass, delta_t):
        self.stiffness = stiffness
        self.rest_len = rest_len
        self.vert = vert
        self.NCONS = stiffness.shape[0]
        self.inv_mass = inv_mass
        self.delta_t = delta_t

    def run(self, x):
        hessian = self.calc_hessian_imply_ti(x)   #taichi impl version
        return hessian

    def calc_hessian_imply_ti(self, x) -> scipy.sparse.csr_matrix:
        assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        NV = x.shape[0]
        
        MAX_NNZ = NCONS* 50     # estimate the nnz: 3*3*4*NCONS

        ii = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        jj = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        vv = np.zeros(dtype=np.float32,shape=MAX_NNZ)


        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
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
        hessian = self.MASS + self.delta_t * self.delta_t * hessian
        return hessian


class CalculateHessianPython():
    def __init__(self, constraintsNew, MASS, delta_t):
        self.constraintsNew = constraintsNew
        self.NCONS = len(constraintsNew)
        self.MASS = MASS
        self.delta_t = delta_t
    
    def run(self, x):
        hessian = self.calc_hessian_imply_py(x)
        return hessian

    def calc_hessian_imply_py(self, x)->scipy.sparse.csr_matrix:
        self.NV = x.shape[0]
        hessian = scipy.sparse.dok_matrix((self.NV*3, self.NV*3),dtype=np.float32)
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                self.EvaluateHessianOneConstraintAttachment(c, x, hessian)
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                self.EvaluateHessianOneConstraintDistance(c, x, hessian)
        hessian = self.MASS + self.delta_t * self.delta_t * hessian
        hessian = hessian.tocsr()
        return hessian
    
    
    def EvaluateHessianOneConstraintDistance(self, c, x, hessian):
        p1 = c.p1
        p2 = c.p2
        x_ij = x[p1] - x[p2]
        l_ij = np.linalg.norm(x_ij)
        l0 = c.rest_len
        ks = c.stiffness
        k = ks * (np.eye(3) - l0/l_ij*(np.eye(3) - np.outer(x_ij, x_ij)/(l_ij*l_ij)))
        for row in range(3):
            for col in range(3):
                val = k[row, col]
                hessian[p1*3+row, p1*3+col] += val
                hessian[p1*3+row, p2*3+col] += -val
                hessian[p2*3+row, p1*3+col] += -val
                hessian[p2*3+row, p2*3+col] += val
        return hessian

    def EvaluateHessianOneConstraintAttachment(self, c, x, hessian):
        # from constraint number j to point number i
        i0 = c.p0
        g = c.stiffness
        for k in range(3):
            hessian[i0*3+k, i0*3+k] += g
        return hessian


class CalculateGradientTaichi():
    def __init__(self, stiffness, rest_len, vert, inv_mass, delta_t, inertial_y):
        self.stiffness = stiffness
        self.rest_len = rest_len
        self.vert = vert
        self.inv_mass = inv_mass
        self.delta_t = delta_t
        self.inertial_y = inertial_y
        self.NCONS = stiffness.shape[0]
        self.NV = inertial_y.shape[0]


    def run(self, x):
        gradient = self.calc_gradient_imply_ti(x)
        return gradient
    

    def calc_gradient_imply_ti(self, x):
        assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        self.NV = x.shape[0]
        
        gradient = np.zeros((self.NV, 3), dtype=np.float32)

        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
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
                gradient[i0] += g_ij
                gradient[i1] -= g_ij
                
        kernel(x,vert,rest_len, NCONS, gradient, stiffness)
        gradient = gradient.flatten()
        h_square = self.delta_t * self.delta_t
        x_tilde = self.inertial_y
        gradient = self.MASS @ (x.flatten() - x_tilde.flatten()) + h_square * gradient
        return gradient


class CalculateGradientPython():
    def __init__(self, constraintsNew, MASS, delta_t, inertial_y):
        self.constraintsNew = constraintsNew
        self.inertial_y = inertial_y
        self.MASS = MASS
        self.delta_t = delta_t

    def run(self, x):
        gradient = self.calc_gradient_imply_py(x)
        return gradient
    
    def calc_gradient_imply_py(self, x):
        self.NV = x.shape[0]
        gradient = np.zeros((self.NV* 3), dtype=np.float32)
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                self.EvaluateGradientOneConstraintAttachment(c, x, gradient.reshape(-1,3))
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                self.EvaluateGradientOneConstraintDistance(c, x, gradient.reshape(-1,3))
        h_square = self.delta_t * self.delta_t
        x_tilde = self.inertial_y
        gradient = self.MASS @ (x.flatten() - x_tilde.flatten()) + h_square * gradient
        return gradient


    def EvaluateGradientOneConstraintAttachment(self, c, x, gradient):
        assert x.shape[1] == 3
        assert gradient.shape[1] == 3
        # from constraint number j to point number i
        i0 = c.p0
        g = c.stiffness * (x[i0] - c.fixed_point)
        gradient[i0] += g

    def EvaluateGradientOneConstraintDistance(self, c, x, gradient):
        assert x.shape[1] == 3
        assert gradient.shape[1] == 3
        # from constraint number j to point number i
        i0 = c.p1
        i1 = c.p2
        x_ij = x[i0] - x[i1]
        g_ij = c.stiffness * (norm(x_ij) - c.rest_len) * normalize(x_ij)
        gradient[i0] += g_ij
        gradient[i1] -= g_ij

