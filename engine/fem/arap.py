import taichi as ti
import meshtaichi_patcher as patcher
from taichi.lang.ops import sqrt
import taichi.math as tm
import numpy as np
import scipy.io as sio
from engine.fem.read_tet import read_tet_mesh
from engine.fem.mesh import Mesh
from result import result_path
from engine.metadata import meta
# ti.init(ti.cuda, kernel_profiler=True, debug=True)
# ti.init(ti.gpu)

dt = 0.001  # timestep size
omega = 0.2  # SOR factor

gravity = ti.Vector([0.0, -9.8, 0.0])
MaxIte = 2
numSubsteps = 10

compute_energy, write_energy_to_file = True, True
show_coarse, show_fine = True, False
frame = ti.field(int, ())

mesh = Mesh(model_name="data/models/bunny1000_2000/bunny1k")

#read restriction operator
P = sio.mmread("data/models/bunny1000_2000/P.mtx")
fine_mesh = Mesh(model_name="data/models/bunny1000_2000/bunny2k", direct_import_faces=True)

def coarse_to_fine():
    coarse_pos = mesh.mesh.verts.pos.to_numpy()
    fine_pos = P @ coarse_pos
    fine_mesh.mesh.verts.pos.from_numpy(fine_pos)

# ---------------------------------------------------------------------------- #
#                                    核心计算步骤                                #
# ---------------------------------------------------------------------------- #

@ti.kernel
def preSolve(dt_: ti.f32):
    # semi-Euler update pos & vel
    for v in mesh.mesh.verts:
        if (v.invMass != 0.0):
            v.vel = v.vel + dt_ * gravity
            v.prevPos = v.pos
            v.pos = v.pos + dt_ * v.vel
            v.predictPos = v.pos


@ti.func
def make_matrix(x, y, z):
    return ti.Matrix([[x, 0, 0, y, 0, 0, z, 0, 0], [0, x, 0, 0, y, 0, 0, z, 0],
                      [0, 0, x, 0, 0, y, 0, 0, z]])


@ti.func
def computeGradient(B, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 + (S[2, 2] - 1)**2)
    # if sumSigma < 1.0e-6:
    #     isSuccess = False

    # (dcdS00, dcdS11, dcdS22)
    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1, S[2, 2] - 1])
    # Compute (dFdx)^T
    neg_sum_col1 = -B[0, 0] - B[1, 0] - B[2, 0]
    neg_sum_col2 = -B[0, 1] - B[1, 1] - B[2, 1]
    neg_sum_col3 = -B[0, 2] - B[1, 2] - B[2, 2]
    dFdp0T = make_matrix(neg_sum_col1, neg_sum_col2, neg_sum_col3)
    dFdp1T = make_matrix(B[0, 0], B[0, 1], B[0, 2])
    dFdp2T = make_matrix(B[1, 0], B[1, 1], B[1, 2])
    dFdp3T = make_matrix(B[2, 0], B[2, 1], B[2, 2])
    # Compute (dsdF)
    u00, u01, u02 = U[0, 0], U[0, 1], U[0, 2]
    u10, u11, u12 = U[1, 0], U[1, 1], U[1, 2]
    u20, u21, u22 = U[2, 0], U[2, 1], U[2, 2]
    v00, v01, v02 = V[0, 0], V[0, 1], V[0, 2]
    v10, v11, v12 = V[1, 0], V[1, 1], V[1, 2]
    v20, v21, v22 = V[2, 0], V[2, 1], V[2, 2]
    dsdF00 = ti.Vector([u00 * v00, u01 * v01, u02 * v02])
    dsdF10 = ti.Vector([u10 * v00, u11 * v01, u12 * v02])
    dsdF20 = ti.Vector([u20 * v00, u21 * v01, u22 * v02])
    dsdF01 = ti.Vector([u00 * v10, u01 * v11, u02 * v12])
    dsdF11 = ti.Vector([u10 * v10, u11 * v11, u12 * v12])
    dsdF21 = ti.Vector([u20 * v10, u21 * v11, u22 * v12])
    dsdF02 = ti.Vector([u00 * v20, u01 * v21, u02 * v22])
    dsdF12 = ti.Vector([u10 * v20, u11 * v21, u12 * v22])
    dsdF22 = ti.Vector([u20 * v20, u21 * v21, u22 * v22])

    # Compute (dcdF)
    dcdF = ti.Vector([
        dsdF00.dot(dcdS),
        dsdF10.dot(dcdS),
        dsdF20.dot(dcdS),
        dsdF01.dot(dcdS),
        dsdF11.dot(dcdS),
        dsdF21.dot(dcdS),
        dsdF02.dot(dcdS),
        dsdF12.dot(dcdS),
        dsdF22.dot(dcdS)
    ])
    g0 = dFdp0T @ dcdF
    g1 = dFdp1T @ dcdF
    g2 = dFdp2T @ dcdF
    g3 = dFdp3T @ dcdF
    return g0, g1, g2, g3, isSuccess


@ti.kernel
def project_fem():
    for c in mesh.mesh.cells:
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
        c.F = D_s @ c.B
        U, S, V = ti.svd(c.F)
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)

        g0, g1, g2, g3, isSuccess = computeGradient(c.B, U, S, V)

        l = p0.invMass * g0.norm_sqr() + p1.invMass * g1.norm_sqr() + p2.invMass * g2.norm_sqr() + p3.invMass * g3.norm_sqr()
        c.dLambda = -(constraint + c.alpha * c.lagrangian) / (
            l + c.alpha)
        c.lagrangian = c.lagrangian + c.dLambda
        c.grad0, c.grad1, c.grad2, c.grad3 = g0, g1, g2, g3

    # for c in mesh.mesh.cells:
        c.verts[0].pos += omega * c.verts[0].invMass * c.dLambda * c.grad0
        c.verts[1].pos += omega * c.verts[1].invMass * c.dLambda * c.grad1
        c.verts[2].pos += omega * c.verts[2].invMass * c.dLambda * c.grad2
        c.verts[3].pos += omega * c.verts[3].invMass * c.dLambda * c.grad3


@ti.kernel
def compute_potential_energy():
    mesh.potential_energy[None] = 0.0
    for c in mesh.mesh.cells:
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
        c.F = D_s @ c.B
        U, S, V = ti.svd(c.F)
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)
        invAlpha = mesh.inv_lame * c.invVol
        mesh.potential_energy[None] += 0.5 * invAlpha *  constraint ** 2 

@ti.kernel
def compute_inertial_energy():
    mesh.inertial_energy[None] = 0.0
    for v in mesh.mesh.verts:
        mesh.inertial_energy[None] += 0.5 / v.invMass * (v.pos - v.predictPos).norm_sqr() * meta.inv_h2


@ti.kernel
def update_pos():
    for c in mesh.mesh.cells:
        c.verts[0].pos += omega * c.verts[0].invMass * c.dLambda * c.grad0
        c.verts[1].pos += omega * c.verts[1].invMass * c.dLambda * c.grad1
        c.verts[2].pos += omega * c.verts[2].invMass * c.dLambda * c.grad2
        c.verts[3].pos += omega * c.verts[3].invMass * c.dLambda * c.grad3



@ti.kernel
def collsion_response():
    for v in mesh.mesh.verts:
        if v.pos[1] < -3.0:
            v.pos[1] = -3.0

@ti.kernel
def postSolve(dt_: ti.f32):
    for v in mesh.mesh.verts:
        if v.invMass != 0.0:
            v.vel = (v.pos - v.prevPos) / dt_


def substep():
    preSolve(dt/numSubsteps)
    mesh.mesh.cells.lagrangian.fill(0.0)
    for ite in range(MaxIte):
        project_fem()
        # update_pos()
        collsion_response()
    postSolve(dt/numSubsteps)

    if compute_energy:
        log_energy()

    frame[None] += 1
    
def log_energy():
    compute_potential_energy()
    compute_inertial_energy()
    mesh.total_energy[None] = mesh.potential_energy[None] + mesh.inertial_energy[None]

    if write_energy_to_file and frame[None]%100==0:
        print(f"frame: {frame[None]} potential: {mesh.potential_energy[None]:.3e} kinetic: {mesh.inertial_energy[None]:.3e} total: {mesh.total_energy[None]:.3e}")
        with open(result_path+"/totalEnergy.txt", "ab") as f:
            np.savetxt(f, np.array([mesh.total_energy[None]]), fmt="%.4e", delimiter="\t")
        with open(result_path+"/potentialEnergy.txt", "ab") as f:
            np.savetxt(f, np.array([mesh.potential_energy[None]]), fmt="%.4e", delimiter="\t")
        with open(result_path+"/kineticEnergy.txt", "ab") as f:
            np.savetxt(f, np.array([mesh.inertial_energy[None]]), fmt="%.4e", delimiter="\t")

def debug(field):
    field_np = field.to_numpy()
    print("---------------------")
    print("name: ", field._name )
    print("shape: ",field_np.shape)
    print("min, max: ", field_np.min(), field_np.max())
    print(field_np)
    print("---------------------")
    np.savetxt("debug_my.txt", field_np.flatten(), fmt="%.4f", delimiter="\t")
    return field_np