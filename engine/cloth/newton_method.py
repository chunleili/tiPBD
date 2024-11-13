import numpy as np
import taichi as ti
import taichi.math as tm
import scipy.sparse
from engine.solver.direct_solver import DirectSolver
from engine.util import timeit, norm_sqr
from engine.cloth.cloth3d import Cloth
from engine.util import set_mass_matrix

class NewNewtonMethod(Cloth):
    def __init__(self,args):
        super().__init__(args, )

        self.EPSILON = 1e-15

        def get_A():
            return self.hessian
        self.linear_solver = DirectSolver(get_A)

        physdata = self.physdata
        stiffness = physdata.stiffness
        rest_len = physdata.rest_len
        vert = physdata.vert
        delta_t = physdata.delta_t
        force = physdata.force
        self.pos = physdata.pos
        self.vel = physdata.vel

        self.NV = physdata.NV
        self.NCONS = physdata.NCONS

        self.NVERTS_ONE_CONS = vert.shape[1]
        predict_pos = self.pos.copy()

        MASS = set_mass_matrix(physdata.mass)
        
        self.calc_hessian_imply_ti = CalculateHessianTaichi(stiffness, rest_len, vert, MASS, delta_t).run
        self.calc_gradient_imply_ti = CalculateGradientTaichi(stiffness, rest_len, vert, MASS, delta_t,  force, predict_pos).run
        self.calc_obj_func_imply_ti = CalculateObjectiveFunctionTaichi(vert, rest_len, stiffness, self.NCONS, MASS, delta_t, force, predict_pos).run

        from engine.line_search import LineSearch
        self.ls = LineSearch(self.calc_obj_func_imply_ti, use_line_search=True, ls_alpha=0.25, ls_beta=0.1, ls_step_size=1.0, ΕPSILON=1e-15)
        self.line_search = self.ls.line_search


    @timeit
    def evaluateGradient(self, x):
        return self.calc_gradient_imply_ti(x)  

    @timeit
    def evaluateHessian(self, x):
        return self.calc_hessian_imply_ti(x)

    def calc_predict_pos(self):
        self.predict_pos = (self.pos + self.delta_t * self.vel)

    def update_pos_and_vel(self,new_pos):
        self.vel = (new_pos - self.pos) / self.delta_t
        self.pos = new_pos.copy()

    @timeit
    def substep_newton(self):
        self.calc_predict_pos()
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

        x += descent_dir.reshape(-1,3) * step_size

        if step_size < self.EPSILON:
            print(f'step_size {step_size} <EPSILON')
            return True
        else:
            return False
        

class CalculateObjectiveFunctionTaichi():
    def __init__(self, vert, rest_len, stiffness, NCONS, MASS, delta_t, force, predict_pos):
        self.vert = vert
        self.rest_len = rest_len
        self.stiffness = stiffness
        self.NCONS = NCONS
        self.MASS = MASS
        self.delta_t = delta_t
        self.force = force
        self.predict_pos = predict_pos

    def run(self, x):
        return self.calc_obj_func_imply_ti(x)

    def calc_obj_func_imply_ti(self,x) -> float:
        @ti.kernel
        def kernel(x:ti.types.ndarray(dtype=tm.vec3),
                   vert:ti.types.ndarray(dtype=tm.ivec2),
                   rest_len:ti.types.ndarray(),
                   NCONS:ti.i32,
                   stiffness:ti.types.ndarray(),
                   )->ti.f32:
            potential = 0.0
            for i in range(NCONS):
                x_ij = x[vert[i][0]] - x[vert[i][1]]
                l_ij = (x_ij).norm()
                l0 = rest_len[i]
                potential += 0.5 * stiffness[i] * (l_ij - l0) ** 2
            return potential
        
        potential_term = kernel(x,self.vert,self.rest_len, self.NCONS, self.stiffness)

        potential_term -= x.flatten()@ self.force.flatten()

        x_diff = x.flatten() - self.predict_pos.flatten()
        inertia_term = 0.5 * x_diff.transpose() @ self.MASS @ x_diff

        h_square = self.delta_t * self.delta_t
        # res = inertia_term + potential_term * h_square #fast mass spring
        res = inertia_term/h_square + potential_term
        return res     
    
class CalculateGradientTaichi():
    def __init__(self, stiffness, rest_len, vert, MASS, delta_t, force, predict_pos):
        self.stiffness = stiffness
        self.rest_len = rest_len
        self.vert = vert
        self.MASS = MASS
        self.delta_t = delta_t
        self.force = force
        self.predict_pos = predict_pos
        self.NCONS = stiffness.shape[0]
        self.NV = predict_pos.shape[0]


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
                   vert:ti.types.ndarray(dtype=tm.ivec2),
                   rest_len:ti.types.ndarray(),
                   NCONS:ti.i32,
                   gradient:ti.types.ndarray(dtype=tm.vec3),
                   stiffness:ti.types.ndarray()
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
        gradient -= self.force.flatten()
        h_square = self.delta_t * self.delta_t
        x_tilde = self.predict_pos
        gradient = self.MASS @ (x.flatten() - x_tilde.flatten()) + h_square * gradient
        return gradient

class CalculateHessianTaichi():
    def __init__(self, stiffness, rest_len, vert, MASS, delta_t):
        self.stiffness = stiffness
        self.rest_len = rest_len
        self.vert = vert
        self.NCONS = stiffness.shape[0]
        self.MASS = MASS
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
                   vert:ti.types.ndarray(dtype=tm.ivec2),
                   rest_len:ti.types.ndarray(),
                   stiffness: ti.types.ndarray(),
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