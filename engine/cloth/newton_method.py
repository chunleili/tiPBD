import numpy as np
import scipy.sparse
import time
import sys, os

sys.path.append(os.getcwd())

from engine.solver.direct_solver import DirectSolver


class SpmatTriplets:
    def __init__(self, n):
        self.n = n
        self.ii = None
        self.jj = None
        self.vv = None


from enum import Enum
class ConstraintType(Enum):
    ATTACHMENT = 0
    DISTANCE = 1

class Constraint:
    def __init__(self, type, edge, rest_len):
        self.type = type
        self.edge = edge
        self.rest_len = rest_len

class NewtonMethod:
    def __init__(self, ist):
        self.NCONS = ist.NCONS
        self.NV = ist.NV
        self.delta_t = ist.delta_t
        self.rest_len = ist.rest_len
        self.edge = ist.edge
        self.gradient = np.zeros(self.NV*3, dtype=np.float32)
        self.descent_dir = np.zeros(self.NV*3, dtype=np.float32)
        self.EPSILON = 1e-6

        def get_A():
            return self.hessian
        self.linear_solver = DirectSolver(get_A)

        self.gradient = np.zeros(self.NV*3, dtype=np.float32)
        self.hessian = scipy.sparse.csr_matrix((self.NV*3, self.NV*3), dtype=np.float32)

        self.use_line_search = True
        self.ls_beta = 0.1
        self.ls_alpha = 0.25

    # https://github.com/chunleili/fast_mass_spring/blob/a203b39ae8f5ec295c242789fe8488dfb7c42951/fast_mass_spring/source/simulation.cpp#L510
    # integrateNewtonDescentOneIteration
    def step_one_iter(self, x):
        #  evaluate gradient direction
        self.evaluateGradient(x, self.gradient)
        if self.gradient.norm_sqr() < self.EPSILON:
            return True
        
        self.evaluateHessian(x, self.hessian)
        self.descent_dir = -self.linear_solver.run(self.gradient)
        step_size = self.line_search(x, self.gradient, self.descent_dir)
        x += self.descent_dir * step_size
        #  report convergence
        if step_size < self.EPSILON:
            return True
        else:
            return False
        

    def evaluateGradient(self, x, gradient):
        gradient.fill(0)

        # springs
        for j in range(self.NCONS):
            if self.constraints[j].type == ConstraintType.ATTACHMENT:
                self.EvaluateGradientOneConstraintAttachment(j, x, gradient)
            else:
                self.EvaluateGradientOneConstraintDistance(j, x, gradient)

        # external forces
        gradient -= self.external_force
        h_square = self.delta_t * self.delta_t
        gradient = self.M @ (x - self.predicted_pos) + h_square * gradient


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
        h_triplets = SpmatTriplets() ## ii jj vv
        for j in range(self.NCONS):
            if self.constraints[j].type == ConstraintType.ATTACHMENT:
                self.EvaluateHessianOneConstraintAttachment(j, x, h_triplets)
            else:
                self.EvaluateHessianOneConstraintDistance(j, x, h_triplets)
        hessian = scipy.sparse.csr_matrix((h_triplets.vv, (h_triplets.ii, h_triplets.jj)), shape=(self.NV*3, self.NV*3))
        hessian = self.M + self.delta_t * self.delta_t * hessian


    def EvaluateHessianOneConstraintAttachment(self, j, x, h_triplets):
        # from constraint number j to point number i
        i0 = self.edge[j][0]
        i1 = self.edge[j][1]
        g = self.stiffness
        for k in range(3):
            h_triplets.ii.append(i0*3+k)
            h_triplets.jj.append(i0*3+k)
            h_triplets.vv.append(g)
            h_triplets.ii.append(i1*3+k)
            h_triplets.jj.append(i1*3+k)
            h_triplets.vv.append(g)
    
    def EvaluateHessianOneConstraintDistance(self, j, x, h_triplets):
        # from constraint number j to point number i
        i0 = self.edge[j][0]
        i1 = self.edge[j][1]
        x_ij = x[i0] - x[i1]
        l_ij = x_ij.norm()
        l0 = self.rest_len[j]
        ks = self.stiffness
        k = ks * (np.eye(3) - l0/l_ij*(np.eye(3) - np.outer(x_ij, x_ij)/(l_ij*l_ij)))
        for row in range(3):
            for col in range(3):
                val = k[row, col]
                h_triplets.ii.append(i0*3+row)
                h_triplets.jj.append(i0*3+col)
                h_triplets.vv.append(val)
                h_triplets.ii.append(i0*3+row)
                h_triplets.jj.append(i1*3+col)
                h_triplets.vv.append(-val)
                h_triplets.ii.append(i1*3+row)
                h_triplets.jj.append(i0*3+col)
                h_triplets.vv.append(-val)
                h_triplets.ii.append(i1*3+row)
                h_triplets.jj.append(i1*3+col)
                h_triplets.vv.append(val)

        
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
        inertia_term = 0.5 * (x - self.inertia_y).transpose() @ self.M @ (x - self.inertia_y)
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
