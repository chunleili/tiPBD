import numpy as np
import scipy.sparse
import time
import sys, os
import taichi as ti

sys.path.append(os.getcwd())

from engine.solver.direct_solver import DirectSolver
from engine.cloth.cloth3d import Cloth
from engine.cloth.constraints import SetupConstraints


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
        # with open("constraints.txt", "a") as f:
        #     for i in self.constraintsNew:
        #         print(i,file=f)

        self.set_mass()
        self.set_external_force(self.args.gravity)

        def get_A():
            return self.hessian
        self.linear_solver = DirectSolver(get_A)

        self.use_line_search = True
        self.ls_beta = 0.1
        self.ls_alpha = 0.25

    def substep_newton(self):
        self.semi_euler()
        self.r_iter.calc_r0()
        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter(self.pos.to_numpy())
            if converge:
                break
        self.n_outer_all.append(self.ite+1)
        self.update_vel()
        self.old_pos.from_numpy(self.pos.to_numpy())

    # https://github.com/chunleili/fast_mass_spring/blob/a203b39ae8f5ec295c242789fe8488dfb7c42951/fast_mass_spring/source/simulation.cpp#L510
    # integrateNewtonDescentOneIteration
    def step_one_iter(self, x):
        #  evaluate gradient direction
        self.gradient = self.evaluateGradient(x, self.gradient)
        if np.linalg.norm(self.gradient)**2 < self.EPSILON:
            return True
        
        self.hessian = self.evaluateHessian(x, self.hessian)
        self.descent_dir = -self.linear_solver.run(self.gradient)
        step_size = self.line_search(x, self.gradient, self.descent_dir)
        x += self.descent_dir * step_size
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
        
        
    def set_external_force(self, gravity):
        gravity_constant = np.array([0, -100, 0]) #FIXME
        self.external_force = np.tile(gravity_constant, self.NV)
        self.external_force = self.MASS @ self.external_force
        

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


    def EvaluateGradientOneConstraintAttachment(self, j, x, gradient):
        # from constraint number j to point number i
        i0 = self.edge[j][0]
        i1 = self.edge[j][1]
        g = self.stiffness * (x[i0] - self.fixed_point[i0])
        gradient[i0] += g
        gradient[i1] -= g

    def EvaluateGradientOneConstraintDistance(self, j, x, gradient):
        # from constraint number j to point number i
        i0 = self.edge[j][0]
        i1 = self.edge[j][1]
        x_ij = x[i0] - x[i1]
        g_ij = self.stiffness * (x_ij.norm() - self.rest_len[j]) * x_ij.normalized()
        gradient[i0] += g_ij
        gradient[i1] -= g_ij


    def evaluateHessian(self, x, hessian):
        # springs
        self.hessian = scipy.sparse.dok_matrix((self.NV*3, self.NV*3),dtype=np.float32)
        self.hessian = self.EvaluateHessianImpl(x, self.hessian)
        scipy.io.mmwrite('hessian1.mtx', self.hessian)
        hessian = self.MASS + self.delta_t * self.delta_t * self.hessian
        return hessian
    

    def EvaluateHessianImpl(self, x, hessian):
        # from constraint number j to point number i
        for c in self.constraintsNew:
            if c.type == "attachment":
                self.EvaluateHessianOneConstraintAttachment(c, x, hessian)
            elif c.type == "stretch" or c.type == "bending":
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
            return 1.0
        else:
            t = 1.0/self.ls_beta
            currentObjectiveValue = self.evaluateObjectiveFunction(x)
            while lhs >= rhs and t > self.EPSILON:
                t *= self.ls_beta
                x_plus_tdx = x + t*descent_dir
                lhs = self.evaluateObjectiveFunction(x_plus_tdx)
                rhs = currentObjectiveValue + self.ls_alpha * t * np.dot(gradient_dir, descent_dir)
            if t < self.EPSILON:
                t = 0.0
            return t
    
    def evaluateObjectiveFunction(self, x):
        potential_term = 0.0
        for j in range(self.NCONS):
            if self.constraints[j].type == ConstraintType.ATTACHMENT:
                potential_term += self.EvaluatePotentialEnergyAttachment(j, x)
            else:
                potential_term += self.EvaluatePotentialEnergyDistance(j, x)
        potential_term -= x @ self.external_force
        inertia_term = 0.5 * (x - self.inertia_y).transpose() @ self.MASS @ (x - self.inertia_y)
        h_square = self.delta_t * self.delta_t
        return inertia_term + potential_term * h_square
    
    def EvaluatePotentialEnergyAttachment(self, j, x):
        i0 = self.edge[j][0]
        i1 = self.edge[j][1]
        return 0.5 * self.stiffness * (x[i0] - self.fixed_point[i0]).norm_sqr() + 0.5 * self.stiffness * (x[i1] - self.fixed_point[i1]).norm_sqr()
    
    def EvaluatePotentialEnergyDistance(self, j, x):
        i0 = self.edge[j][0]
        i1 = self.edge[j][1]
        x_ij = x[i0] - x[i1]
        l_ij = x_ij.norm()
        l0 = self.rest_len[j]
        return 0.5 * self.stiffness * (l_ij - l0) * (l_ij - l0)




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

