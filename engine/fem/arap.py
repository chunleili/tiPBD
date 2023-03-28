import taichi as ti
from taichi.lang.ops import sqrt
import taichi.math as tm
import numpy as np
import scipy.io as sio
from engine.fem.mesh import Mesh
from engine.metadata import meta
from engine.log import log_energy


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
            v.vel = v.vel + dt_ * meta.gravity
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
def project_constraints():
    for c in mesh.mesh.cells:
        p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
        D_s = ti.Matrix.cols([p1.pos - p0.pos, p2.pos - p0.pos, p3.pos - p0.pos])
        c.F = D_s @ c.B
        U, S, V = ti.svd(c.F)
        c.fem_constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2 +(S[2, 2] - 1)**2)

        g0, g1, g2, g3, isSuccess = computeGradient(c.B, U, S, V)
        c.grad0, c.grad1, c.grad2, c.grad3 = g0, g1, g2, g3

        l = p0.invMass * g0.norm_sqr() + p1.invMass * g1.norm_sqr() + p2.invMass * g2.norm_sqr() + p3.invMass * g3.norm_sqr()
        c.dLambda = -(c.fem_constraint + c.alpha * c.lagrangian) / (
            l + c.alpha)
        c.lagrangian = c.lagrangian + c.dLambda


@ti.kernel
def update_pos():
    for c in mesh.mesh.cells:
        c.verts[0].pos += meta.relax_factor * c.verts[0].invMass * c.dLambda * c.grad0
        c.verts[1].pos += meta.relax_factor * c.verts[1].invMass * c.dLambda * c.grad1
        c.verts[2].pos += meta.relax_factor * c.verts[2].invMass * c.dLambda * c.grad2
        c.verts[3].pos += meta.relax_factor * c.verts[3].invMass * c.dLambda * c.grad3
    # for c in mesh.mesh.cells:
    #     pass
    #     c.verts[0].pos += meta.relax_factor * c.verts[0].invMass * c.get_member_field["verts"].grad[0]
    #     c.verts[1].pos += meta.relax_factor * c.verts[1].invMass * c.get_member_field["verts"].grad[1]
    #     c.verts[2].pos += meta.relax_factor * c.verts[2].invMass * c.get_member_field["verts"].grad[2]
    #     c.verts[3].pos += meta.relax_factor * c.verts[3].invMass * c.get_member_field["verts"].grad[3]

@ti.kernel
def compute_potential_energy():
    mesh.potential_energy[None] = 0.0
    for c in mesh.mesh.cells:
        invAlpha = meta.inv_lame_lambda * c.inv_vol
        mesh.potential_energy[None] += 0.5 * invAlpha *  c.fem_constraint ** 2 

@ti.kernel
def compute_inertial_energy(): 
    mesh.inertial_energy[None] = 0.0
    for v in mesh.mesh.verts:
        mesh.inertial_energy[None] += 0.5 / v.invMass * (v.pos - v.predictPos).norm_sqr() * meta.inv_h2


@ti.kernel
def collsion_response():
    for v in mesh.mesh.verts:
        if v.pos[1] < meta.ground.y:
            v.pos[1] = meta.ground.y

@ti.kernel
def postSolve(dt_: ti.f32):
    for v in mesh.mesh.verts:
        if v.invMass != 0.0:
            v.vel = (v.pos - v.prevPos) / dt_

def substep():
    preSolve(meta.dt/meta.num_substeps)
    mesh.mesh.cells.lagrangian.fill(0.0)
    for ite in range(meta.max_iter):
        project_constraints()
        update_pos()
        collsion_response()
    postSolve(meta.dt/meta.num_substeps)

    if meta.compute_energy:
        compute_potential_energy()
        # with ti.ad.Tape(loss=mesh.potential_energy[None]):
        #     compute_potential_energy()
        compute_inertial_energy()
        mesh.total_energy[None] = mesh.potential_energy[None] + mesh.inertial_energy[None]
        log_energy(mesh)
    meta.frame += 1