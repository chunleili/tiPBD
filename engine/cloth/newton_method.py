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
    def __init__(self, args, extlib=None):
        super().__init__(args, extlib)
        self.EPSILON = 1e-6

        self.set_constraints_from_read()
        self.read_data_from_fms()

        self.hessian = None

        self.use_line_search = True
        self.ls_alpha = 0.25
        self.ls_beta = 0.1
        self.ls_step_size = 1.0

        self.args.smoother_type = "jacobi"
        self.args.smoother_niter = 3
        self.args.maxiter_Axb = 300
        self.linsol = AmgCuda(self.args, self.extlib)

    def set_constraints_from_read(self):
        """
        Set constraints from constraints.txt
        
        Data to be set:
        - NCONS : number of constraints
        - vert: array of vertex indices of constraints
        - stiffness: array of stiffness of constraints
        - rest_len: array of rest length of constraints
        - NVERTS_ONE_CONS: number of verts per constraints, e.g., 2 for distance
        - pin(shape=NV): bool field indicating which is pin point
        - pinpos(shape=NV): field of pin positions(points other than pin points will be omitted)
        - pinlist: list to indicate the vertex indices of pin points
        - pinposlist: list of positions of pined pos
        - cType(shape=NCONS): constraint type
        - p0(shape=NCONS): pin vertex index for attachment constraint
        - fixed_points(shape=NCONS): fixed points positions for attachment constraints
        """
        from engine.cloth.constraints import constraintsAdapter, SetupConstraints

        s = SetupConstraints(
            self.pos.to_numpy(), self.edge.to_numpy()
        ).read_constraints("E:/Dev/fast_mass_spring/fast_mass_spring/constraints.txt")
        ad = constraintsAdapter(s)
        self.NCONS = ad.NCONS
        logging.info(f"    read constraints NCONS: {self.NCONS}")
        self.constraints = ad.val
        self.rest_len = ad.rest_len
        # CAUTION: vert may not be the same with edge! And NCONS != NE
        self.vert = ad.vert
        self.stiffness = ad.stiffness
        self.NVERTS_ONE_CONS = ad.NVERTS_ONE_CONS

        self.pinlist = ad.pinlist
        self.pinposlist = ad.pinposlist
        from engine.util import pinlist_to_field
        self.pin, self.pinpos = pinlist_to_field(self.pinlist,self.pinposlist,self.NV)
        ...

        self.cType = ad.cType
        self.p0 = ad.p0
        self.fixed_point = ad.fixed_point


    @staticmethod
    def set_stiffness_field(alpha_tilde, delta_t):
        stiffness = ti.field(dtype=ti.f32, shape=alpha_tilde.shape[0])

        @ti.kernel
        def kernel(stiffness: ti.template(), alpha_tilde: ti.template()):
            for i in alpha_tilde:
                stiffness[i] = 1.0 / (alpha_tilde[i] * delta_t**2)

        kernel(stiffness, alpha_tilde)
        return stiffness
    

    def set_stiffness_matrix(self):
        NCONS = self.NCONS
        stiffness = self.set_stiffness_field(self.alpha_tilde, self.delta_t)
        self.stiffness_matrix = scipy.sparse.dia_matrix((stiffness.to_numpy(), [0]), shape=(NCONS, NCONS), dtype=np.float32)
        assert self.stiffness_matrix.shape[0]==self.NCONS
        assert self.stiffness_matrix.shape[1]==self.NCONS
        

    def set_mass(self):
        from engine.util import set_mass_matrix_from_invmass
        self.MASS = set_mass_matrix_from_invmass(self.inv_mass)

        # from engine.util import set_inv_mass_from_mass_matrix
        # set_inv_mass_from_mass_matrix(self.inv_mass)

    def read_data_from_fms(self):
        os.chdir("E:/Dev/fast_mass_spring/fast_mass_spring/")
        # x1 = mmread("x.mtx").toarray().reshape(-1, 3)
        # self.pos.from_numpy(x1)
        self.MASS = mmread("MASS.mtx")
        # self.predict_pos.from_numpy(mmread("y.mtx").toarray().reshape(-1, 3))
        self.force = mmread("f.mtx").toarray().reshape(-1, 3)
        self.NV = self.MASS.shape[0]//3
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
        self.pos_old.from_numpy(self.pos.to_numpy())
        self.calc_force()
        self.pos.from_numpy(self.predict_pos.to_numpy())
        for self.ite in range(self.args.maxiter):
            converge = self.step_one_iter_newton(self.pos)
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
    def step_one_iter_newton(self, x):
        print(f"ite: {self.ite}")
        gradient = self.evaluateGradient(x)
        nrmsqr = norm_sqr(gradient)
        if nrmsqr < self.EPSILON:
            print(f"gradient nrmsqr {nrmsqr} <EPSILON")
            return True

        self.hessian = self.evaluateHessian(x)

        descent_dir,r_Axb = self.linsol.run_v2(self.hessian, gradient.flatten())
        descent_dir = -descent_dir.reshape(-1, 3).astype(np.float32)

        step_size = self.line_search(x, self.predict_pos, gradient, descent_dir)

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
        # print(f"    energy:{res:.8e}")
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
                g_ij = stiffness[i] * (dis) * x_ij.normalized()
                gradient[i0] += g_ij
                gradient[i1] -= g_ij

        kernel(x, vert, rest_len, gradient, stiffness)
        # gradient -= force
        gradient = (
            MASS @ (x.to_numpy().flatten() - x_tilde.to_numpy().flatten())
            + delta_t * delta_t * gradient.flatten()
        )
        gradient = gradient.reshape(-1, 3)

        return gradient
    

    def calc_gradient_imply_constraint(self, pos, C=None, G=None):
        """
        Calculate the energy gradient by assembling the constraints
        
        # gradE = 1/(dt^2) M (x-x_tilde) + nabla C^T K C
        gradE = M (x-x_tilde) + nabla C^T K C * dt^2

        where 

        - K is diagonal stiffness matrix,

        - vector C is in R^m. For one constraint j:

            C_j = l_ij - l_0

        - sparse matrix nabla C is in R^(m x 3n). For one constraint j:

            nabla C_j = [g, -g], where g = (p-q)/l is 3x1 vector.

            One constraint j corresponds to one row of sparse matrix nabla C.

            The first nonzero value g locates at j row, 3*i1/3*i1+1/3*i1+2 columns;

            The second nonzero value -g locates at j row, 3*i2/3*i2+1/3*i2+2 columns

        """
        if C is None or G is None:
            from engine.cloth.C_and_gradC_distance import compute_C_and_gradC_distance, fill_G_distance
            vert = self.vert
            rest_len = self.rest_len
            NV = self.NV

            C, gradC = compute_C_and_gradC_distance(pos,vert,rest_len)
            C = C.to_numpy()
            G = fill_G_distance(gradC, vert, NV)
            
        
        MASS = self.MASS
        ppos = self.predict_pos
        delta_t = self.delta_t
        stiffness_matrix = self.stiffness_matrix

        inertia_term =  MASS @ (pos.to_numpy().flatten() - ppos.to_numpy().flatten())
        potential_term = G.T @ stiffness_matrix @ C  * delta_t**2
        gradE = inertia_term + potential_term
        assert gradE.shape[0] == 3*NV
        assert not np.isnan(gradE).any(), "gradE contains NaN values"
        return gradE.reshape(-1,3)
    

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








class CompareNewtonMethod(NewtonMethod):
    def __init__(self, args, extlib=None):
        Cloth.__init__(self, args, extlib)
        self.set_mass()
        self.vert=self.edge
        self.NCONS = self.NE
        self.EPSILON = 1e-6
        self.use_line_search = True
        self.ls_alpha = 0.25
        self.ls_beta = 0.1
        self.ls_step_size = 1.0

    def calc_energy(self, pos, predict_pos):
        constraints = self.update_constraints(pos)

        @ti.kernel
        def compute_potential_energy_kernel(
            constraints: ti.template(),
            alpha_tilde: ti.template(),
            delta_t: ti.f32,
        )->ti.f32:
            potential_energy = 0.0
            for i in range(constraints.shape[0]):
                inv_alpha =  1.0 / (alpha_tilde[i]*delta_t**2)
                potential_energy += 0.5 * inv_alpha * constraints[i]**2
            return potential_energy

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

        self.potential_energy = compute_potential_energy_kernel(constraints, self.alpha_tilde, self.delta_t)
        self.inertial_energy = compute_inertial_energy_kernel(pos, predict_pos, self.inv_mass, self.delta_t)
        self.energy = self.potential_energy + self.inertial_energy
        return self.energy

    @timeit
    def substep_newton(self):
        self.calc_predict_pos()
        self.calc_force()
        self.pos.from_numpy(self.predict_pos.to_numpy())
        for self.ite in range(self.args.maxiter):
            # converge = self.compare_oneiter_newton_mgpbd()
            converge = self.test_calc_gradient_imply_constraint()
            # converge = self.step_one_iter_newton(self.pos)
            if converge:
                break
        self.n_outer_all.append(self.ite + 1)
        self.update_vel()

    @staticmethod
    @ti.kernel
    def compute_C_and_gradC_kernel(self,
        pos:ti.template(),
        gradC: ti.template(),
        vert:ti.template(),
        constraints:ti.template(),
        rest_len:ti.template(),
    ):
        for i in range(vert.shape[0]):
            idx0, idx1 = vert[i]
            dis = pos[idx0] - pos[idx1]
            lij = dis.norm()
            if lij == 0.0:
                continue
            constraints[i] = lij - rest_len[i]
            if constraints[i] < 1e-6:
                continue
            g = dis.normalized()

            gradC[i, 0] += g
            gradC[i, 1] += -g



    def fill_G(self, gradC):
        vert = self.vert
        NV = self.NV
        NCONS = vert.shape[0]

        MAX_NNZ =NCONS * 6

        @staticmethod
        @ti.kernel
        def fill_gradC_triplets_kernel(
            ii:ti.types.ndarray(dtype=ti.i32),
            jj:ti.types.ndarray(dtype=ti.i32),
            vv:ti.types.ndarray(dtype=ti.f32),
            gradC: ti.template(),
            vert: ti.template(),
        ):
            cnt=0
            ti.loop_config(serialize=True)
            for j in range(vert.shape[0]):
                ind = vert[j]
                for p in range(2):
                    for d in range(3):
                        i = ind[p]
                        ii[cnt],jj[cnt],vv[cnt] = j, 3 * i + d, gradC[j, p][d]
                        cnt+=1
        
        G_ii, G_jj, G_vv = np.zeros(MAX_NNZ, dtype=np.int32), np.zeros(MAX_NNZ, dtype=np.int32), np.zeros(MAX_NNZ, dtype=np.float32)
        assert not np.any(np.isnan(gradC.to_numpy()))
        assert not np.any(np.isnan(G_vv))
        fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, vert)
        G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(NCONS, 3*NV))
        return G


    def fill_A_by_spmm(self, M_inv, ALPHA_TILDE, gradC):
        G = self.fill_G(gradC)
        A = G @ M_inv @ G.transpose() + ALPHA_TILDE
        A = scipy.sparse.csr_matrix(A)
        return A
    

    # def set_Minv_and_ALPHA_TILDE(self):
    #     MASS = self.MASS
    #     stiffness = self.stiffness
    #     delta_t = self.delta_t
        
    #     def set_Minv_from_MASS(MASS):
    #         invmass = 1.0 / MASS.diagonal()
    #         np.where(np.isinf(invmass), 0.0, invmass)
    #         M_inv = scipy.sparse.diags(invmass)
    #         return M_inv
        
    #     def set_ALPHA_TILDE_from_stiffness(stiffness, delta_t):
    #         alpha_tilde = ti.field(dtype=ti.f32, shape=stiffness.shape[0])
    #         if isinstance(stiffness, np.ndarray):
    #             s = ti.field(dtype=ti.f32, shape=stiffness.shape)
    #             s.from_numpy(stiffness)
    #             stiffness = s

    #         @ti.kernel
    #         def kernel(stiffness: ti.template(), alpha_tilde: ti.template()):
    #             for i in alpha_tilde:
    #                 alpha_tilde[i] = 1.0 / stiffness[i] / delta_t**2
    #         kernel(stiffness, alpha_tilde)
            
    #         ALPHA_TILDE = scipy.sparse.diags(alpha_tilde.to_numpy())
    #         return ALPHA_TILDE, alpha_tilde

    #     M_inv = set_Minv_from_MASS(MASS)
    #     ALPHA_TILDE, alpha_tilde = set_ALPHA_TILDE_from_stiffness(stiffness, delta_t)

    #     self.M_inv = M_inv
    #     self.ALPHA_TILDE = ALPHA_TILDE
    #     self.alpha_tilde = alpha_tilde

    #     return M_inv, ALPHA_TILDE


    def assemble_A(self, gradC):
        A = self.fill_A_by_spmm(self.M_inv, self.ALPHA_TILDE, gradC)
        assert not np.any(np.isnan(A.data))
        return A
    

    def compute_b(self, constraints):
        alpha_tilde = self.alpha_tilde
        lagrangian = self.lagrangian
        assert alpha_tilde.shape == lagrangian.shape
        b = -constraints.to_numpy()-alpha_tilde.to_numpy()*lagrangian.to_numpy()
        return b
    
    def compute_C_and_gradC(self, pos):
        vert = self.vert    
        rest_len = self.rest_len
        self.NVERTS_ONE_CONS = 2
        constraints = ti.field(dtype=ti.f32, shape=vert.shape[0])
        gradC = ti.Vector.field(3, dtype=ti.f32, shape=(vert.shape[0],self.NVERTS_ONE_CONS))
        self.compute_C_and_gradC_kernel(pos,
                                        gradC,
                                        vert,
                                        constraints,
                                        rest_len)
        assert not np.any(np.isnan(constraints.to_numpy()))
        assert not np.any(np.isnan(gradC.to_numpy()))
        return constraints, gradC

    def dlam2dpos(self, dlam, gradC):
        vert = self.vert
        inv_mass = self.inv_mass

        dLambda = ti.field(dtype=ti.f32, shape=dlam.shape[0])
        dLambda.from_numpy(dlam)
        dpos = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)

        @ti.kernel
        def dlam2dpos_kernel(
            vert:ti.template(),
            inv_mass:ti.template(),
            dLambda:ti.template(),
            gradC:ti.template(),
            dpos:ti.template(),
        ):
            for i in range(dpos.shape[0]):
                dpos[i] = ti.Vector([0.0, 0.0, 0.0])
            
            for i in range(vert.shape[0]):
                idx0, idx1 = vert[i]
                invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
                gradient = gradC[i, 0]
                if invM0 != 0.0:
                    dpos[idx0] += invM0 * dLambda[i] * gradient
                if invM1 != 0.0:
                    dpos[idx1] -= invM1 * dLambda[i] * gradient

        dlam2dpos_kernel(vert, inv_mass,
                         dLambda, gradC, dpos)
        
        return dpos
    
    def update_pos(self, dpos, pos):
        omega = self.args.omega
        inv_mass = self.inv_mass

        @ti.kernel
        def update_pos_kernel(
            inv_mass:ti.template(),
            dpos:ti.template(),
            pos:ti.template(),
            omega:ti.f32,
        ):
            for i in range(inv_mass.shape[0]):
                if inv_mass[i] != 0.0:
                    pos[i] += omega * dpos[i]
        update_pos_kernel(inv_mass, dpos, pos, omega)
        self.pos = pos
        return pos


    def update_lambda(self, dlambda, lagrangian):
        @ti.kernel
        def add_lam_kernel(dlambda:ti.types.ndarray(),
                           lagrangian:ti.template()):
            for i in range(lagrangian.shape[0]):
                lagrangian[i] += dlambda[i]
        
        add_lam_kernel(dlambda, lagrangian)
        self.lagrangian = lagrangian 
        return lagrangian
    

    def step_one_iter_mgpbd(self, pos, lagrangian):
        constraints, gradC = self.compute_C_and_gradC(pos)
        b = self.compute_b(constraints)

        nrm = np.linalg.norm(b)
        logging.info(f"    b norm: {nrm:.8e}")
        if nrm < self.EPSILON:
            logging.info(f"converge because b norm < EPSILON")
            return True
        
        A = self.assemble_A(gradC)

        dlambda,r_Axb = self.linsol.run_v2(A, b)
        dlambda = dlambda.astype(np.float32)

        dpos = self.dlam2dpos(dlambda, gradC)
        self.update_lambda(dlambda, lagrangian)
        self.update_pos(dpos, pos)
        
        return pos, lagrangian
    

    def update_constraints(self, pos):
        vert = self.vert
        rest_len = self.rest_len
        constraints = ti.field(dtype=ti.f32, shape=self.NCONS)

        @ti.kernel
        def update_constraints_kernel(
            pos:ti.template(),
            vert:ti.template(),
            rest_len:ti.template(),
            constraints:ti.template(),
        ):
            for i in range(vert.shape[0]):
                idx0, idx1 = vert[i]
                dis = pos[idx0] - pos[idx1]
                constraints[i] = dis.norm() - rest_len[i]
        update_constraints_kernel(pos, vert, rest_len, constraints)
        return constraints
    
    def calc_C_norm(self, pos):
        C = self.update_constraints(pos)
        from engine.util import calc_norm
        nrm = calc_norm(C)
        return nrm


    def calc_strain(self, pos)->float:
        vert = self.vert
        rest_len = self.rest_len
        strain = ti.field(dtype=ti.f32, shape=self.vert.shape[0])
        @ti.kernel
        def calc_strain_cloth_kernel(
            vert:ti.template(),
            rest_len:ti.template(),
            pos:ti.template(),
            strain:ti.template(),
        ):
            for i in range(vert.shape[0]):
                idx0, idx1 = vert[i]
                dis = pos[idx0] - pos[idx1]
                l = dis.norm()
                if l< 1e-6:
                    continue
                strain[i] = (l - rest_len[i])/rest_len[i]
        calc_strain_cloth_kernel(vert, rest_len, pos, strain)
        s = np.max(strain.to_numpy())
        return s
    
        

    def compare_oneiter_newton_mgpbd(self):
        from engine.ti_kernels import init_scale
        init_scale(self.predict_pos, 1.5)
        self.pos.from_numpy(self.predict_pos.to_numpy())

        self.force = np.zeros((self.NV, 3), dtype=np.float32)   

        self.lagrangian = ti.field(dtype=ti.f32, shape=self.NCONS)

        maxiter = 10
        self.args.omega = 1.0

        energy0 = self.calc_energy(self.pos, self.predict_pos)
        logging.info(f"    initial energy0: {energy0:.8e}")


        def calc_res(strains, Cs, energies, pos, predict_pos):
            s = self.calc_strain(pos)
            strains.append(s)
            logging.info(f"    strain: {s:.8e}")

            c = self.calc_C_norm(pos)
            Cs.append(c)
            logging.info(f" constraints: {c}")

            e = self.calc_energy(pos, predict_pos)
            energies.append(e)
            logging.info(f"    energy: {e:.8e}")
            

        # mgpbd
        self.pos.from_numpy(self.predict_pos.to_numpy())
        mgpbd_energies = [energy0]
        mgpbd_strains = [self.calc_strain(self.pos)]
        mgpbd_C = [self.calc_C_norm(self.pos)]
        self.lagrangian.fill(0)
        for i in range(maxiter):
            self.step_one_iter_mgpbd(self.pos, self.lagrangian)
            calc_res(mgpbd_strains, mgpbd_C, mgpbd_energies, self.pos, self.predict_pos)


        self.pos.from_numpy(self.predict_pos.to_numpy())
        self.lagrangian.fill(0)
        xpbd_energies = [energy0]
        xpbd_strains = [self.calc_strain(self.pos)]
        xpbd_C = [self.calc_C_norm(self.pos)]
        for i in range(maxiter):
            self.project_constraints_xpbd()
            self.update_pos(self.dpos, self.pos)
            calc_res(xpbd_strains, xpbd_C, xpbd_energies, self.pos, self.predict_pos)


        # newton
        self.pos.from_numpy(self.predict_pos.to_numpy())
        newton_energies = [energy0]
        newton_strains = [self.calc_strain(self.pos)]
        newton_C = [self.calc_C_norm(self.pos)]
        for i in range(maxiter):
            self.step_one_iter_newton(self.pos)
            calc_res(newton_strains, newton_C, newton_energies, self.pos, self.predict_pos)
            

        logging.info(f"Newton vs mgpbd vs xpbd")
        logging.info("energy-------------------")
        for i in range(maxiter):
            logging.info(f"{i}: {newton_energies[i]:.6e} vs {mgpbd_energies[i]:.6e} vs {xpbd_energies[i]:.6e}")

        logging.info("strain-------------------")
        for i in range(maxiter):
            logging.info(f"{i}: {newton_strains[i]:.6e} vs {mgpbd_strains[i]:.6e} vs {xpbd_strains[i]:.6e}")
        
        logging.info("C-------------------")
        for i in range(maxiter):
            logging.info(f"{i}: {newton_C[i]:.6e} vs {mgpbd_C[i]:.6e} vs {xpbd_C[i]:.6e}")



        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,1)
        axs.plot(newton_strains, label="newton")
        axs.plot(mgpbd_strains, label="mgpbd")
        axs.plot(xpbd_strains, label="xpbd")
        axs.legend()
        axs.set_title("strain")
        # plt.show()

        fig,axs = plt.subplots(1,1)
        axs.plot(newton_energies, label="newton")
        axs.plot(mgpbd_energies, label="mgpbd")
        axs.plot(xpbd_energies, label="xpbd")
        axs.legend()
        axs.set_title("energy")
        # plt.show()


        fig,axs = plt.subplots(1,1)
        axs.plot(newton_C, label="newton")
        axs.plot(mgpbd_C, label="mgpbd")
        axs.plot(xpbd_C, label="xpbd")
        axs.legend()
        axs.set_title("constraint")
        plt.show()




class TestNewtonMethod(NewtonMethod):
    def __init__(self, args, extlib=None):
        Cloth.__init__(self,args, extlib)
        self.args = args
    
    def substep_newton(self):
        from engine.file_utils import do_restart 
        do_restart(self.args, self)
        self.test_calc_gradient_imply_constraint()


    def test_calc_gradient_imply_constraint(self):
        x = self.pos
        self.vert = self.edge
        self.set_mass()
        gradE1 = self.calc_gradient_imply_constraint(x)
        gradE2 = self.calc_gradient_cloth_imply_ti(x)
        from engine.util import vec_is_equal
        assert vec_is_equal(gradE1, gradE2, 1e-6)
        ...