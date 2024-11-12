import numpy as np
import scipy.io
import scipy.sparse
import time
import sys, os
import taichi as ti
import taichi.math as tm

sys.path.append(os.getcwd())

from engine.solver.direct_solver import DirectSolver
from engine.cloth.cloth3d import Cloth
from engine.cloth.constraints import SetupConstraints, ConstraintType
from engine.util import timeit, norm, norm_sqr, normalize

def debug(x, name='vec'):  
    print(f'{name}: {x.shape}')
    norm = np.linalg.norm(x)
    max_val = np.max(x)
    amax = np.argmax(x)
    min_val = np.min(x)
    amin = np.argmin(x)
    print(f'    norm: {norm} max_val: {max_val}, amax: {amax} min_val: {min_val}, amin: {amin}\n')
    np.savetxt(f'{name}.txt', x)

def debugmat(x, name='mat'):  
    print(f'{name}: {x.shape}')
    norm = np.linalg.norm(x.data)
    max_val = np.max(x.data)
    min_val = np.min(x.data)
    print(f'    norm: {norm} max_val: {max_val}  min_val: {min_val}\n')
    scipy.io.mmwrite(f"{name}.mtx", x)



@ti.data_oriented
class NewtonMethod(Cloth):
    def __init__(self,args):
        super().__init__(args)
        
        self.pos = self.pos.to_numpy()
        self.predict_pos = self.predict_pos.to_numpy()
        self.vel = self.vel.to_numpy()

        self.EPSILON = 1e-15

        self.setupConstraints = SetupConstraints(self.pos, self.edge.to_numpy(), self.args)
        self.constraintsNew = self.setupConstraints.constraints
        self.adapter = self.setupConstraints.adapter

        self.set_mass()

        def get_A():
            return self.hessian
        self.linear_solver = DirectSolver(get_A)

        self.use_line_search = True
        self.ls_beta = 0.1
        self.ls_alpha = 0.25
        self.ls_step_size = 1.0


    def calc_predict_pos(self):
        self.predict_pos = (self.pos + self.delta_t * self.vel)
    
    def update_pos_and_vel(self,new_pos):
        self.vel = (new_pos - self.pos) / self.delta_t
        self.pos = new_pos.copy()

    @timeit
    def substep_newton(self):
        self.calc_predict_pos()
        self.calc_external_force(self.args.gravity)
        pos_next = self.predict_pos.copy()

        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter(pos_next)
            if converge:
                break
        self.n_outer_all.append(self.ite+1)
        self.update_pos_and_vel(pos_next)
        

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

        descent_dir,_ = self.linear_solver.run(gradient)
        descent_dir = -descent_dir

        step_size = self.line_search(x, gradient, descent_dir)
        print(f"energy: {self.total_energy}")

        x += descent_dir.reshape(-1,3) * step_size

        if step_size < self.EPSILON:
            print(f'step_size {step_size} <EPSILON')
            return True
        else:
            return False
        
    def set_mass(self):
        # pmass = 1.0 / self.NV
        pmass = 1.0
        self.MASS = scipy.sparse.diags([pmass]*self.NV*3)
        self.M_inv = scipy.sparse.diags([1.0/pmass]*self.NV*3)
        
        
    def calc_external_force(self, gravity=[0,-9.8,0]):
        # gravity = [0,-100,0] fast mass spring
        self.external_force = np.zeros(self.NV*3, dtype=np.float32)
        gravity_constant = np.array(gravity) #FIXME
        ext = np.tile(gravity_constant, self.NV)
        self.external_acc = ext.copy().reshape(-1,3)
        self.external_force = self.MASS @ ext
        

    @timeit
    def evaluateGradient(self, x):
        gradient = np.zeros((self.NV* 3), dtype=np.float32)
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                self.EvaluateGradientOneConstraintAttachment(c, x, gradient.reshape(-1,3))
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                self.EvaluateGradientOneConstraintDistance(c, x, gradient.reshape(-1,3))
        gradient -= self.external_force
        h_square = self.delta_t * self.delta_t
        x_tilde = self.predict_pos
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

    @timeit
    def evaluateHessian(self, x):
        # hessian = self.calc_hessian_imply_py(x) #python impl version
        hessian = self.calc_hessian_imply_ti(x)   #taichi impl version
        return hessian

    def calc_hessian_imply_py(self, x)->scipy.sparse.csr_matrix:
        hessian = scipy.sparse.dok_matrix((self.NV*3, self.NV*3),dtype=np.float32)
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                self.EvaluateHessianOneConstraintAttachment(c, x, hessian)
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                self.EvaluateHessianOneConstraintDistance(c, x, hessian)
        hessian = self.MASS + self.delta_t * self.delta_t * hessian
        hessian = hessian.tocsr()
        return hessian
    

    def calc_hessian_imply_ti(self, x) -> scipy.sparse.csr_matrix:
        assert x.shape[1]==3
        stiffness = self.adapter.stiffness
        rest_len = self.adapter.rest_len
        vert = self.adapter.vert
        NCONS = self.adapter.NCONS
        
        MAX_NNZ = NCONS* 50     # estimate the nnz: 3*3*4*NCONS

        ii = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        jj = np.zeros(dtype=np.int32,  shape=MAX_NNZ)
        vv = np.zeros(dtype=np.float32,shape=MAX_NNZ)


        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
                   vert:ti.template(),
                   rest_len:ti.template(),
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
        kernel(x,vert,rest_len, NCONS, ii, jj, vv)
        hessian = scipy.sparse.coo_matrix((vv,(ii,jj)),shape=(self.NV*3, self.NV*3),dtype=np.float32)
        hessian = self.MASS + self.delta_t * self.delta_t * hessian
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
        
        
    def line_search(self, x, gradient_dir, descent_dir):
        if not self.use_line_search:
            return self.ls_step_size
        
        t = 1.0/self.ls_beta
        currentObjectiveValue = self.evaluateObjectiveFunction(x)
        ls_times = 0
        while ls_times==0 or (lhs >= rhs and t > self.EPSILON):
            t *= self.ls_beta
            x_plus_tdx = (x.flatten() + t*descent_dir).reshape(-1,3)
            lhs = self.evaluateObjectiveFunction(x_plus_tdx)
            rhs = currentObjectiveValue + self.ls_alpha * t * np.dot(gradient_dir, descent_dir)
            ls_times += 1
        self.total_energy = lhs
        print(f'ls_times: {ls_times}')

        if t < self.EPSILON:
            t = 0.0
        else:
            self.ls_step_size = t
        return t
    
    def evaluateObjectiveFunction(self, x):
        energy1 = self.calc_obj_func_imply_py(x)
        # energy2 = self.calc_obj_func_imply_ti()
        return energy1
    

    def calc_obj_func_imply_ti(self) -> float:
        return super().calc_total_energy()
    

    def calc_obj_func_imply_py(self, x):
        potential_term = 0.0
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                potential_term += self.EvaluatePotentialEnergyAttachment(c, x.reshape(-1,3))
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                potential_term += self.EvaluatePotentialEnergyDistance(c, x.reshape(-1,3))

        potential_term -= x.flatten()@ self.external_force

        x_diff = x.flatten() - self.predict_pos.flatten()
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
    
    def calc_total_energy(self):
        return super().calc_total_energy()