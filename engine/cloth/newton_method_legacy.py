"""This is for debugging fast mass spring by tiantian liu"""

import numpy as np
import scipy.io
import scipy.sparse
import time
import sys, os
import taichi as ti
import taichi.math as tm
import logging
from scipy.io import mmwrite, mmread

sys.path.append(os.getcwd())

from engine.solver.amg_cuda import AmgCuda
from engine.solver.direct_solver import DirectSolver
from engine.cloth.cloth3d import Cloth
from engine.util import timeit, norm_sqr


@ti.data_oriented
class NewtonMethod(Cloth):
    def __init__(self, args, extlib):
        super().__init__(args)
        self.EPSILON = 1e-6
        # self.set_mass()
        # self.stiffness = self.set_stiffness(self.alpha_tilde, self.delta_t)
        # self.vert =self.edge

        self.set_constraints_from_read()
        self.read_data_from_fms()

        self.linsol = DirectSolver(lambda: self.hessian)

        self.use_line_search = True
        self.ls_alpha = 0.25
        self.ls_beta = 0.1
        self.ls_step_size = 1.0

    def set_constraints_from_read(self):
        from engine.cloth.constraints import constraintsAdapter, SetupConstraints

        s = SetupConstraints(
            self.pos.to_numpy(), self.edge.to_numpy()
        ).read_constraints("constraints.txt")
        ad = constraintsAdapter(s)
        self.NCONS = ad.NCONS
        self.constraints = ad.val
        self.rest_len = ad.rest_len
        self.vert = ad.vert
        self.stiffness = ad.stiffness

        self.pinlist = ad.pinlist
        self.pinposlist = ad.pinposlist
        from engine.util import pinlist_to_field
        self.pin, self.pinpos = pinlist_to_field(self.pinlist,self.pinposlist,self.NV)
        ...

        self.cType = ad.cType
        self.p0 = ad.p0
        self.fixed_point = ad.fixed_point

    def set_stiffness(self, alpha_tilde, delta_t):
        stiffness = ti.field(dtype=ti.f32, shape=alpha_tilde.shape[0])

        @ti.kernel
        def kernel(stiffness: ti.template(), alpha_tilde: ti.template()):
            for i in alpha_tilde:
                stiffness[i] = 1.0 / (alpha_tilde[i] * delta_t**2)

        kernel(stiffness, alpha_tilde)
        return stiffness

    def set_mass(self):
        from engine.util import set_mass_matrix_from_invmass
        self.MASS = set_mass_matrix_from_invmass(self.inv_mass)

        # from engine.util import set_inv_mass_from_mass_matrix
        # set_inv_mass_from_mass_matrix(self.inv_mass)

    def read_data_from_fms(self):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        x1 = mmread("x.mtx").toarray().reshape(-1, 3)
        self.pos.from_numpy(x1)
        self.MASS = mmread("MASS.mtx")
        self.predict_pos.from_numpy(mmread("y.mtx").toarray().reshape(-1, 3))
        self.force = mmread("f.mtx").toarray().reshape(-1, 3)
        os.chdir(self.prj_path)

    @timeit
    def evaluateGradient(self, x):
        gradient = self.calc_gradient_cloth_imply_ti(x)
        return gradient

    @timeit
    def evaluateHessian(self, x):
        hessian = self.calc_hessian_imply_ti(x)
        return hessian

    def debug_gradient(self, x):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        x1 = mmread("x.mtx").toarray().reshape(-1, 3)
        self.MASS = mmread("MASS.mtx")
        x.from_numpy(x1)
        self.predict_pos.from_numpy(mmread("y.mtx").toarray().reshape(-1, 3))
        self.force = mmread("f.mtx").toarray().reshape(-1, 3)
        g1 = mmread("g1.mtx").toarray().reshape(-1, 3)
        os.chdir(self.prj_path)

        g2 = self.calc_gradient_cloth_imply_ti(x)
        print((g1-g2).max())
        assert np.allclose(g1, g2)
        print("gradient ok")
        return g2

    def debug_hessian(self, x):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        x1 = mmread("x.mtx").toarray().reshape(-1, 3)
        x.from_numpy(x1)
        self.predict_pos.from_numpy(mmread("y.mtx").toarray().reshape(-1, 3))

        h1 = mmread("h1.mtx")
        h2 = self.calc_hessian_imply_ti(x)
        from engine.util import csr_is_equal
        assert csr_is_equal(h1, h2)
        os.chdir(self.prj_path)
        print("hessian ok")

        return h2
    
    def calc_predict_pos(self):
        self.predict_pos.from_numpy(self.pos.to_numpy() + self.vel.to_numpy() * self.delta_t)

    def calc_force(self):
        MASS = self.MASS
        f = np.zeros((self.NV, 3), dtype=np.float32)

        gravity_constant = 100.0
        f[:, 1] = -gravity_constant
        f = MASS @ f.flatten()
        f = f.reshape(-1, 3)

        self.force = f


    @timeit
    def substep_newton(self):
        self.calc_predict_pos()
        self.calc_force()
        self.pos.from_numpy(self.predict_pos.to_numpy())
        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter(self.pos)
            if converge:
                break
        self.n_outer_all.append(self.ite + 1)
        self.update_vel()

    def debug_solve(self,gradient):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        # h1 = mmread("h1.mtx")
        # g1 = mmread("g1.mtx")
        d1 = mmread("d1.mtx").toarray().flatten()
        os.chdir(self.prj_path)
        descent_dir = scipy.sparse.linalg.spsolve(self.hessian, gradient.flatten())
        d2 = -descent_dir
        assert np.allclose(d1, d2, atol=1e-6), (d1-d2).max()
        print("solve ok")
        ...

    def debug_x(self, x):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        x1 = mmread("x.mtx").toarray().reshape(-1, 3)
        os.chdir(self.prj_path)
        assert np.allclose(x1, x.to_numpy())
        print(f"{(x1- x.to_numpy()).max()}")
        print("x ok")
        ...
    
    def debug_predict_pos(self):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        y1 = mmread("y.mtx").toarray().reshape(-1, 3)
        os.chdir(self.prj_path)
        print((y1-self.predict_pos.to_numpy()).max())
        assert np.allclose(y1, self.predict_pos.to_numpy())
        print("predict_pos ok")
        ...


    # https://github.com/chunleili/fast_mass_spring/blob/a203b39ae8f5ec295c242789fe8488dfb7c42951/fast_mass_spring/source/simulation.cpp#L510
    # integrateNewtonDescentOneIteration
    def step_one_iter(self, x):
        print(f"ite: {self.ite}")
        gradient = self.evaluateGradient(x)
        nrmsqr = norm_sqr(gradient)
        if nrmsqr < self.EPSILON:
            print(f"gradient nrmsqr {nrmsqr} <EPSILON")
            return True

        self.hessian = self.evaluateHessian(x)

        descent_dir, r_Axb = self.linsol.run(gradient.flatten())
        descent_dir = -descent_dir.reshape(-1, 3).astype(np.float32)
        logging.info(f"    r_Axb: {r_Axb[0]:.2e} {r_Axb[-1]:.2e}")

        step_size = self.line_search(x, self.predict_pos, gradient, descent_dir)
        logging.info(f"    step_size: {step_size:.2e}")

        x.from_numpy(x.to_numpy() + descent_dir.reshape(-1, 3) * step_size)

        if step_size < self.EPSILON:
            print(f"step_size {step_size} <EPSILON")
            return True
        else:
            return False

    def debug_energy(self, x, predict_pos):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        x1 = mmread("x.mtx").toarray().reshape(-1, 3)
        x.from_numpy(x1)
        self.MASS = mmread("MASS.mtx")
        self.predict_pos.from_numpy(mmread("y.mtx").toarray().reshape(-1, 3))
        self.force = mmread("f.mtx").toarray().reshape(-1, 3)
        e1 = np.loadtxt("energy.txt")
        os.chdir(self.prj_path)
        e2 = self.calc_energy(x,predict_pos)
        print("e1-e2",e1-e2)
        assert abs(e1-e2)<1e-6, e1-e2
        print("energy ok")


    def debug_inertial(self, x, i2):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        i1 = np.loadtxt("inertia_term.txt")
        os.chdir(self.prj_path)

        print("i1-p2",i1-i2)
        assert abs(i1-i2)<1e-6, i1-i2
        ...

    def debug_potential(self,x):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        x.from_numpy(mmread("x.mtx").toarray().reshape(-1, 3))
        p1 = np.loadtxt("potential_term.txt")
        os.chdir(self.prj_path)

        p2 = self.calc_potential(x)
        print("p1-p2",p1-p2)
        assert abs(p1-p2)<1e-6, p1-p2
        print("potential ok")
        ...

    # w/o external force
    def calc_potential(self, x):
        vert = self.vert
        stiffness = self.stiffness
        rest_len = self.rest_len
        cType = self.cType
        fixed_point = self.fixed_point
        p0 = self.p0

        @ti.kernel
        def kernel(
            x: ti.template(),
            vert: ti.template(),
            rest_len: ti.template(),
            stiffness: ti.template(),
            cType:ti.types.ndarray(),
            p0:ti.types.ndarray(),
            fixed_point:ti.types.ndarray(dtype=tm.vec3),
        ) -> ti.f32:
            potential = 0.0
            # ti.loop_config(serialize=True)
            for i in range(stiffness.shape[0]):
                if cType[i] == 1:
                    e = 0.5*stiffness[i] * (x[p0[i]] - fixed_point[i]).norm_sqr()
                    potential += e
                    continue
                x_ij = x[vert[i][0]] - x[vert[i][1]]
                l_ij = (x_ij).norm()
                l0 = rest_len[i]
                potential += 0.5 * stiffness[i] * (l_ij - l0) ** 2
            return potential

        potential_term = kernel(x, vert, rest_len, stiffness, cType, p0, fixed_point)
        return potential_term
    

    def calc_energy(self, x, predict_pos):
        vert = self.vert
        stiffness = self.stiffness
        force = self.force
        rest_len = self.rest_len
        NCONS = self.NCONS
        delta_t = self.delta_t
        MASS = self.MASS
        
        potential_term = self.calc_potential(x)

        potential_term -= x.to_numpy().flatten() @ force.flatten()

        x_diff = x.to_numpy().flatten() - predict_pos.to_numpy().flatten()
        inertia_term = 0.5 * x_diff.transpose() @ MASS @ x_diff

        h_square = delta_t * delta_t
        res = inertia_term + potential_term * h_square #fast mass spring
        # res = inertia_term / h_square + potential_term
        print(f"    energy:{res:.8e}")
        return res

    def calc_gradient_cloth_imply_ti(self, x):
        # assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NV = x.shape[0]
        x_tilde = self.predict_pos
        MASS = self.MASS
        delta_t = self.delta_t
        force = self.force

        gradient = np.zeros((NV, 3), dtype=np.float32)

        @ti.kernel
        def kernel(
            x: ti.template(),
            vert: ti.template(),
            rest_len: ti.template(),
            gradient: ti.types.ndarray(dtype=tm.vec3),
            stiffness: ti.template(),
        ):
            for i in range(rest_len.shape[0]):
                i0, i1 = vert[i]
                x_ij = x[i0] - x[i1]
                l_ij = x_ij.norm()
                if l_ij == 0.0:
                    continue  # prevent nan if x_ij=0 0 0
                dis = l_ij - rest_len[i]
                if dis < 1e-6:
                    continue
                # print(i,"dis", dis)
                g_ij = stiffness[i] * (dis) * x_ij.normalized()
                gradient[i0] += g_ij
                gradient[i1] -= g_ij

        kernel(x, vert, rest_len, gradient, stiffness)
        gradient -= force
        gradient = (
            MASS @ (x.to_numpy().flatten() - x_tilde.to_numpy().flatten())
            + delta_t * delta_t * gradient.flatten()
        )
        gradient = gradient.reshape(-1, 3)

        return gradient
    

    def calc_hessian_imply_ti(self, x) -> scipy.sparse.csr_matrix:
        # assert x.shape[1]==3
        stiffness = self.stiffness
        rest_len = self.rest_len
        vert = self.vert
        NCONS = self.NCONS
        NV = x.shape[0]
        delta_t = self.delta_t
        MASS = self.MASS
        cType = self.cType
        p0 = self.p0

        MAX_NNZ = NCONS * 50  # estimate the nnz: 3*3*4*NCONS

        ii = np.zeros(dtype=np.int32, shape=MAX_NNZ)
        jj = np.zeros(dtype=np.int32, shape=MAX_NNZ)
        vv = np.zeros(dtype=np.float32, shape=MAX_NNZ)



        @ti.kernel
        def kernel(
            x: ti.template(),
            vert: ti.template(),
            rest_len: ti.template(),
            stiffness: ti.template(),
            cType:ti.types.ndarray(),
            p0:ti.types.ndarray(),
            ii: ti.types.ndarray(),
            jj: ti.types.ndarray(),
            vv: ti.types.ndarray(),
        ):
            kk = 0
            ti.loop_config(serialize=True) #CAUTION: this is important
            for i in range(stiffness.shape[0]):
                p1, p2 = vert[i]
                if cType[i] == 1:
                    ks = stiffness[i]
                    for row in ti.static(range(3)):
                        val = stiffness[i]
                        ii[kk] = 3 * p0[i] + row
                        jj[kk] = 3 * p0[i] + row
                        vv[kk] += val
                        kk += 1
                    continue

                x_ij = x[p1] - x[p2]
                l_ij = x_ij.norm()
                if l_ij == 0.0:
                    continue
                l0 = rest_len[i]
                ks = stiffness[i]
                k = ks * (tm.eye(3)- l0/l_ij * (tm.eye(3) - x_ij.outer_product(x_ij) / (l_ij * l_ij)))
                # k = ks * (np.eye(3) - l0/l_ij*(np.eye(3) - np.outer(x_ij, x_ij)/(l_ij*l_ij)))
                for row in ti.static(range(3)):
                    for col in ti.static(range(3)):
                        val = k[row, col]
                        ii[kk] = 3 * p1 + row
                        jj[kk] = 3 * p1 + col
                        vv[kk] += val
                        kk += 1
                        ii[kk] = 3 * p1 + row
                        jj[kk] = 3 * p2 + col
                        vv[kk] += -val
                        kk += 1
                        ii[kk] = 3 * p2 + row
                        jj[kk] = 3 * p1 + col
                        vv[kk] += -val
                        kk += 1
                        ii[kk] = 3 * p2 + row
                        jj[kk] = 3 * p2 + col
                        vv[kk] += val
                        kk += 1

        kernel(x, vert, rest_len, stiffness, cType, p0, ii, jj, vv)
        hessian = scipy.sparse.coo_matrix(
            (vv, (ii, jj)), shape=(NV * 3, NV * 3), dtype=np.float32
        )
        hessian = MASS + (delta_t * delta_t) * hessian
        hessian = hessian.tocsr()
        return hessian

    def line_search(self, x, predict_pos, gradient_dir, descent_dir):
        if not self.use_line_search:
            return self.ls_step_size

        x_plus_tdx = self.copy_field(x)
        descent_dir = descent_dir.reshape(-1, 3)

        t = 1.0 / self.ls_beta
        currentObjectiveValue = self.calc_energy(x, predict_pos)
        ls_times = 0
        while ls_times == 0 or (lhs >= rhs and t > self.EPSILON):
            t *= self.ls_beta
            # x_plus_tdx = (x.flatten() + t*descent_dir).reshape(-1,3)
            self.calc_x_plus_tdx(x_plus_tdx, x, t, descent_dir)
            lhs = self.calc_energy(x_plus_tdx, predict_pos)
            rhs = currentObjectiveValue + self.ls_alpha * t * np.dot(
                gradient_dir.flatten(), descent_dir.flatten()
            )
            ls_times += 1
        self.energy = lhs
        print(f"    energy: {self.energy:.8e}")
        print(f"    ls_times: {ls_times}")

        if t < self.EPSILON:
            t = 0.0
        else:
            self.ls_step_size = t
        return t

    @staticmethod
    def calc_x_plus_tdx(x_plus_tdx, x, t, descent_dir):
        @ti.kernel
        def kernel(
            x_plus_tdx: ti.template(),
            x: ti.template(),
            t: ti.f32,
            descent_dir: ti.types.ndarray(dtype=tm.vec3),
        ):
            for i in range(x_plus_tdx.shape[0]):
                x_plus_tdx[i] = x[i] + t * descent_dir[i]
        kernel(x_plus_tdx, x, t, descent_dir)

    @staticmethod
    def copy_field(f1):
        f2 = ti.Matrix.field(
            n=f1.n, m=f1.m, ndim=f1.ndim, dtype=f1.dtype, shape=f1.shape
        )

        @ti.kernel
        def copy_kernel(f1: ti.template(), f2: ti.template()):
            for i in f1:
                f2[i] = f1[i]

        copy_kernel(f1, f2)
        return f2
