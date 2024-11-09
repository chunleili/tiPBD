import numpy as np
import scipy.sparse
import time
import sys, os
import taichi as ti

sys.path.append(os.getcwd())

from engine.solver.direct_solver import DirectSolver
from engine.cloth.cloth3d import Cloth
from engine.cloth.constraints import SetupConstraints, ConstraintType
from engine.util import timeit

@ti.data_oriented
class NewtonMethod(Cloth):
    def __init__(self):
        super().__init__()
        self.gradient = np.zeros(self.NV*3, dtype=np.float32)
        self.descent_dir = np.zeros(self.NV*3, dtype=np.float32)
        self.hessian = scipy.sparse.csr_matrix((self.NV*3, self.NV*3), dtype=np.float32)
        self.EPSILON = 1e-6
        self.stiffness = 80.0 # FIXME
        self.stiffness_attachment = 120.0 # FIXME
        # self.constraintsType = np.zeros(self.NCONS, dtype=ConstraintType)
        # self.fixed_points = [0,420]
        # self.set_constraints_type()
        setupConstraints = SetupConstraints(self.pos.to_numpy(), self.edge.to_numpy())
        setupConstraints.setup_constraints()
        self.constraintsNew = setupConstraints.constraints


        self.set_mass()

        def get_A():
            return self.hessian
        self.linear_solver = DirectSolver(get_A)

        self.use_line_search = True
        self.ls_beta = 0.1
        self.ls_alpha = 0.25
        self.ls_step_size = 1.0

    def calc_predict_pos(self):
        self.predict_pos.from_numpy(self.pos.to_numpy() + self.delta_t * self.vel.to_numpy())
    

    def substep_newton(self):
        self.calc_predict_pos()
        self.pos_next = self.predict_pos.to_numpy()
        self.calc_external_force(self.args.gravity)

        self.r_iter.calc_r0()
        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter(self.pos_next)
            if converge:
                break
        self.n_outer_all.append(self.ite+1)
        self.update_vel()
        self.pos.from_numpy(self.pos_next)
        self.old_pos.from_numpy(self.pos.to_numpy())

    # https://github.com/chunleili/fast_mass_spring/blob/a203b39ae8f5ec295c242789fe8488dfb7c42951/fast_mass_spring/source/simulation.cpp#L510
    # integrateNewtonDescentOneIteration
    def step_one_iter(self, x):
        #  evaluate gradient direction
        self.gradient = self.evaluateGradient(x, self.gradient)
        if np.linalg.norm(self.gradient)**2 < self.EPSILON:
            return True
        
        self.hessian = self.evaluateHessian(x, self.hessian)
        self.descent_dir,_ = self.linear_solver.run(self.gradient)
        self.descent_dir = -self.descent_dir
        step_size = self.line_search(x, self.gradient, self.descent_dir)
        x += self.descent_dir.reshape(-1,3) * step_size
        #  report convergence
        if step_size < self.EPSILON:
            return True
        else:
            return False
        
    def set_mass(self):
        pmass = 1.0 / self.NV
        self.MASS = scipy.sparse.diags([pmass]*self.NV*3)
        self.M_inv = scipy.sparse.diags([1.0/pmass]*self.NV*3)
        self.inv_mass_np = np.array([1.0/pmass]*self.NV)
        self.inv_mass.from_numpy(self.inv_mass_np)
        # scipy.io.mmwrite('mass.mtx', self.MASS)
        
        
    def calc_external_force(self, gravity):
        self.external_force = np.zeros(self.NV*3, dtype=np.float32)
        gravity_constant = np.array([0, -100, 0]) #FIXME
        ext = np.tile(gravity_constant, self.NV)
        self.external_force = self.MASS @ ext
        

    def evaluateGradient(self, x, gradient):
        gradient.fill(0)
        self.fill_G()
        gradient = self.G.T @ self.ALPHA_inv @ self.constraints.to_numpy()

        gradient -= self.external_force
        h_square = self.delta_t * self.delta_t
        x = self.pos.to_numpy().flatten()
        x_tilde = self.predict_pos.to_numpy().flatten()
        gradient = self.MASS @ (x - x_tilde) + h_square * gradient
        return gradient


    # def EvaluateGradientOneConstraintAttachment(self, j, x, gradient):
    #     # from constraint number j to point number i
    #     i0 = self.edge[j][0]
    #     i1 = self.edge[j][1]
    #     g = self.stiffness * (x[i0] - self.fixed_point[i0])
    #     gradient[i0] += g
    #     gradient[i1] -= g

    # def EvaluateGradientOneConstraintDistance(self, j, x, gradient):
    #     # from constraint number j to point number i
    #     i0 = self.edge[j][0]
    #     i1 = self.edge[j][1]
    #     x_ij = x[i0] - x[i1]
    #     g_ij = self.stiffness * (x_ij.norm() - self.rest_len[j]) * x_ij.normalized()
    #     gradient[i0] += g_ij
    #     gradient[i1] -= g_ij

    @timeit
    def evaluateHessian(self, x, hessian):
        # springs
        self.hessian = scipy.sparse.dok_matrix((self.NV*3, self.NV*3),dtype=np.float32)
        self.hessian = self.EvaluateHessianImpl(x, self.hessian)
        self.hessian = self.MASS + self.delta_t * self.delta_t * self.hessian
        # scipy.io.mmwrite('hessian1.mtx', self.hessian)
        return self.hessian
    

    def EvaluateHessianImpl(self, x, hessian):
        # from constraint number j to point number i
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                self.EvaluateHessianOneConstraintAttachment(c, x, hessian)
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                self.EvaluateHessianOneConstraintDistance(c, x, hessian)
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
            x_plus_tdx = x.flatten() + t*descent_dir
            lhs = self.evaluateObjectiveFunction(x_plus_tdx.reshape(-1,3))
            rhs = currentObjectiveValue + self.ls_alpha * t * np.dot(gradient_dir, descent_dir)
            ls_times += 1

        if t < self.EPSILON:
            t = 0.0
        else:
            self.ls_step_size = t
        return t
    
    def evaluateObjectiveFunction(self, x):
        potential_term = 0.0
        for c in self.constraintsNew:
            if c.type == ConstraintType.ATTACHMENT:
                potential_term += self.EvaluatePotentialEnergyAttachment(c, x)
            elif c.type == ConstraintType.STRETCH or c.type == ConstraintType.ΒENDING:
                potential_term += self.EvaluatePotentialEnergyDistance(c, x)

        x_tilde = self.predict_pos.to_numpy().flatten()

        potential_term -= x.flatten()@ self.external_force

        inertia_term = 0.5 * (x.flatten() - x_tilde).transpose() @ self.MASS @ (x.flatten() - x_tilde)
        h_square = self.delta_t * self.delta_t
        self.total_energy = inertia_term + potential_term * h_square
        return self.total_energy
    
    # // 0.5*k*(current_length)^2
    def EvaluatePotentialEnergyAttachment(self, c, x):
        res = 0.5 * c.stiffness * self.norm_sqr(x[c.p0] - c.fixed_point)
        return res
    
    # // 0.5*k*(current_length - rest_length)^2
    def EvaluatePotentialEnergyDistance(self, c, x):
        x_ij = x[c.p1] - x[c.p2]
        l_ij = self.norm(x_ij)
        l0 = c.rest_len
        res = 0.5 * c.stiffness * (l_ij - l0) ** 2
        return res 

    def norm_sqr(self, x):
        return np.linalg.norm(x)**2
    
    def norm(self, x):
        return np.linalg.norm(x)


    # @ti.kernel
    # def compute_K_kernel(h_triplet:ti.types.ndarray(),
    #                      pos:ti.template(),
    #                      edge:ti.template(),
    #                      lagrangian:ti.template()
    #                      ):
    #     for i in range(edge.shape[0]):
    #         idx0, idx1 = edge[i]
    #         dis = pos[idx0] - pos[idx1]
    #         L= dis.norm()
    #         g = dis.normalized()

    #         #geometric stiffness K: 
    #         # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
    #         # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
    #         # https://blog.csdn.net/weixin_43940314/article/details/139448858
    #         # geometric stiffness
    #         """
    #             k = lambda[i]/l * (I - n * n')
    #             K = | Hessian_{x1,x1}, Hessian_{x1,x2}   |  = | k  -k|
    #                 | Hessian_{x1,x2}, Hessian_{x2,x2}   |    |-k   k|
    #         """
    #         k0 = lagrangian[i] / L * (1 - g[0]*g[0])
    #         k1 = lagrangian[i] / L * (1 - g[1]*g[1])
    #         k2 = lagrangian[i] / L * (1 - g[2]*g[2])
    #         h_triplet[idx0*3]   = k0
    #         h_triplet[idx0*3+1] = k1
    #         h_triplet[idx0*3+2] = k2
    #         h_triplet[idx1*3]   = k0
    #         h_triplet[idx1*3+1] = k1
    #         h_triplet[idx1*3+2] = k2

