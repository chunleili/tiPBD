import scipy
import taichi as ti
from time import perf_counter
import logging
from engine.cloth.cloth3d import Cloth
# parser.add_argument("-use_bending", type=int, default=False)

@ti.data_oriented
class PrimalXPBD(Cloth):
    def __init__(self):
        super().__init__()
        self.Minv_gg = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.dpos_withg  = ti.Vector.field(3, dtype=float, shape=(self.NV))


    # v1: with g, modify b and dpos
    def AMG_PXPBD_v1_dlam2dpos(self, x,G, Minv_gg):
        dLambda_ = x.copy()
        self.lagrangian.from_numpy(self.lagrangian.to_numpy() + dLambda_)
        dpos = self.M_inv @ G.transpose() @ dLambda_ 
        dpos -=  Minv_gg
        dpos = dpos.reshape(-1, 3)
        self.pos.from_numpy(self.pos_mid.to_numpy() + self.omega*dpos)


    # v2: blended, only modify dpos
    def AMG_PXPBD_v2_dlam2dpos(x,self):
        tic = perf_counter()
        self.dLambda.from_numpy(x)
        self.dpos.fill(0)
        self.dpos_withg.fill(0)
        transfer_back_to_pos_mfree_kernel()
        self.update_pos()
        self.compute_C_and_gradC()
        # G = fill_G()
        transfer_back_to_pos_mfree_kernel_withg()
        # dpos_withg_np = (predict_pos.to_numpy() - pos.to_numpy()).flatten() + M_inv @ G.transpose() @ lagrangian.to_numpy()
        # dpos_withg.from_numpy(dpos_withg_np.reshape(-1, 3))
        update_pos_blend(self.inv_mass, self.dpos, self.pos, self.dpos_withg)
        update_pos_kernel(self.inv_mass, self.dpos_withg, self.pos)
        logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")



    def AMG_PXPBD_v1_b(self, G):
        # #we calc inverse mass times gg(primary residual), because NCONS may contains infinity for fixed pin points. And gg always appears with inv_mass.
        self.compute_b()

        # PXPBD_b_kernel(edge, pos, predict_pos, lagrangian, inv_mass, gradC, b, Minv_gg)
        MASS = scipy.sparse.diags(1.0/(self.M_inv.diagonal()+1e-12), format="csr")
        Minv_gg =  MASS@self.M_inv@(self.pos.to_numpy().flatten() - self.predict_pos.to_numpy().flatten()) - self.M_inv @ G.transpose() @ self.lagrangian.to_numpy()
        b += G @ Minv_gg
        return b, Minv_gg

    # v1-mfree
    def PXPBD_v1_mfree_transfer_back_to_pos(self, x, Minv_gg):
        self.dLambda.from_numpy(x)
        self.dpos.fill(0)
        self.PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg)
        self.update_pos_kernel(self.inv_mass, self.dpos, self.pos)


    # v1-mfree
    @ti.kernel
    def PXPBD_v1_mfree_transfer_back_to_pos_kernel(
        edge:ti.template(),
        inv_mass:ti.template(),
        gradC:ti.template(),
        lagrangian:ti.template(),
        dpos:ti.template(),
        dLambda:ti.template(),
        Minv_gg:ti.template()):
        for i in range(edge.shape[0]):
            idx0, idx1 = edge[i]
            invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

            delta_lagrangian = dLambda[i]
            lagrangian[i] += delta_lagrangian

            gradient = gradC[i, 0]
            
            if invM0 != 0.0:
                dpos[idx0] += invM0 * delta_lagrangian * gradient - Minv_gg[idx0]
            if invM1 != 0.0:
                dpos[idx1] -= invM1 * delta_lagrangian * gradient - Minv_gg[idx1]
            


        

    def substep_pxpbd_v1(self, args):
        args.PXPBD_ksi = 1.0
        args.use_PXPBD_v1 = True
        self.semi_euler()
        self.lagrangian.fill(0.0)
        self.r_iter.calc_r0()
        for self.ite in range(args.maxiter):
            tic_iter = perf_counter()
            copy_field(self.pos_mid, self.pos)
            self.compute_C_and_gradC()
            b = self.compute_b()
            G = fill_G()
            b, Minv_gg = self.compute_b(G)
            x, r_Axb = linsol.run(b)
            self.dlam2dpos(x, G, Minv_gg)
            self.r_iter.calc_r(self.frame,self.ite, tic_iter, r_Axb)
            export_mat(self, get_A0_cuda, b)
            logging.info(f"iter time(with export): {(perf_counter()-tic_iter)*1000:.0f}ms")
            if self.r_iter.check():
                break
        tic = perf_counter()
        logging.info(f"n_outer: {self.ite+1}")
        self.n_outer_all.append(self.ite+1)
        self.update_vel()
        logging.info(f"post-loop time: {(perf_counter()-tic)*1000:.0f}ms")

@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]


@ti.kernel
def PXPBD_b_kernel(edge:ti.template(),pos:ti.template(), predict_pos:ti.template(), lagrangian:ti.template(), inv_mass:ti.template(), gradC:ti.template(), b:ti.types.ndarray(), Minv_gg:ti.template()):
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        if invM0 != 0.0:
            Minv_gg[idx0] = invM0 * lagrangian[i] * gradC[i, 0] + (pos[idx0] - predict_pos[idx0])
        if invM1 != 0.0:
            Minv_gg[idx1] = invM1 * lagrangian[i] * gradC[i, 1] + (pos[idx0] - predict_pos[idx0])

    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        if invM1 != 0.0 and invM0 != 0.0:
            b[idx0] += gradC[i, 0] @ Minv_gg[idx0] + gradC[i, 1] @ Minv_gg[idx1]

        #     Minv_gg =  (pos.to_numpy().flatten() - predict_pos.to_numpy().flatten()) - M_inv @ G.transpose() @ lagrangian.to_numpy()
        #     b += G @ Minv_gg

@ti.kernel
def transfer_back_to_pos_mfree_kernel_withg(
    edge:ti.template(),
    inv_mass:ti.template(),
    gradC:ti.template(),
    lagrangian:ti.template(),
    dpos_withg:ti.template(),
    old_pos:ti.template(),
    predict_pos:ti.template(),
):
    for i in range(edge.shape[0]):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        gradient = gradC[i, 0]
        if invM0 != 0.0:
            dpos_withg[idx0] += invM0 * lagrangian[i] * gradient 
        if invM1 != 0.0:
            dpos_withg[idx1] -= invM1 * lagrangian[i] * gradient

    for i in range(inv_mass.shape[0]):
        if inv_mass[i] != 0.0:
            dpos_withg[i] += predict_pos[i] - old_pos[i]


@ti.kernel
def update_pos_blend(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    dpos_withg:ti.template(),
):
    for i in range(pos.shape[0]):
        if inv_mass[i] != 0.0:
            pos[i] += args.omega *((1-args.PXPBD_ksi) * dpos[i] + args.PXPBD_ksi * dpos_withg[i])