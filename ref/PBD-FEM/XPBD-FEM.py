"""
Then we use XPBD-FEM (gpu version, Jacobian solver) to simulate the deformation of 2D object
"""
import taichi as ti
from taichi.lang.ops import sqrt
from readObj import Objfile
import time

ti.init(arch=ti.gpu)

h = 0.003  # timestep size
staticPoint = 8
omega = 0.5  # SOR factor
compliance = 1.0e-4  # Fat Tissuse compliance, for more specific material,please see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h
                      )  # timestep related compliance, see XPBD paper
obj = Objfile()
# obj.read("./2dMesh.obj")
obj.readTxt("models/armadillo.txt")
vertices = obj.getVertice()
triangles = obj.getFaces()

NV = obj.getNumVertice()
NF = obj.getNumFaces()  # number of faces
pos = ti.Vector.field(2, float, NV)
oldPos = ti.Vector.field(2, float, NV)
vel = ti.Vector.field(2, float, NV)  # velocity of particles
invMass = ti.field(float, NV)  #inverse mass of particles


f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # D_m^{-1}
F = ti.Matrix.field(2, 2, float, NF)  # deformation gradient
lagrangian = ti.field(float, NF)  # lagrangian multipliers

gradient = ti.Vector.field(2,float, 3 * NF)
dLambda = ti.field(float, NF)

gravity = ti.Vector([0, -1.2])
MaxIte = 20
NumSteps = 10

# For validation
dualResidual = ti.field(float, ())
primalResidual = ti.field(float, ())

attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.kernel
def init_pos():
    for i in range(NV):
        pos[i] *= ti.Vector([0.4, 0.4])
        pos[i] += ti.Vector([0.1, 0.01])
        oldPos[i] = pos[i]
        vel[i] = ti.Vector([0, 0])
        invMass[i] = 1.0
    for i in range(staticPoint):
        invMass[i] = 0.0
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])
        B[i] = B_i_inv.inverse()


@ti.kernel
def resetLagrangian():
    for i in range(NF):
        lagrangian[i] = 0.0

@ti.func
def vec(x):
    return ti.Vector([x[0,0], x[1,0], x[0,1], x[1, 1]])

@ti.func
def computeGradient_f(idx, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    if sumSigma < 1.0e-6:
        isSuccess = False

    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, 0, 0, S[1, 1] - 1
                                       ])  # (dcdS11, dcdS12, dcdS21, dcdS22)
    # Compute (dFdx)^T
    dFdxT = ti.Vector([[-B[idx][0,0]-B[idx][1,0],                         0, -B[idx][0,1]-B[idx][1,1],            0],
                      [                       0,  -B[idx][0,0]-B[idx][1,0],                        0, -B[idx][0,1]-B[idx][1,1]],
                      [             B[idx][0,0],                         0,              B[idx][0,1],            0],
                      [                       0,               B[idx][0,0],                        0,  B[idx][0,1]],
                      [             B[idx][1,0],                         0,              B[idx][1,1],            0],
                      [                       0,               B[idx][1,0],                        0,  B[idx][1,1]]])
    # Compute (dsdF)
    dF00 = ti.Vector([[1,0], [0, 0]])
    dF10 = ti.Vector([[0,0], [1, 0]])
    dF01 = ti.Vector([[0,1], [0, 0]])
    dF11 = ti.Vector([[0,0], [0, 1]])
    dsdF00 = U.transpose() @ dF00 @ V
    dsdF10 = U.transpose() @ dF10 @ V
    dsdF01 = U.transpose() @ dF01 @ V
    dsdF11 = U.transpose() @ dF11 @ V

    dsdF = ti.Matrix.cols([vec(dsdF00), vec(dsdF10), vec(dsdF01), vec(dsdF11)])

    g = dFdxT @ dsdF.transpose() @ dcdS
    g0 = ti.Vector([g[0], g[1]])

    g1 = ti.Vector([g[2], g[3]])
    g2 = ti.Vector([g[4], g[5]])
    return g0, g1, g2, isSuccess


@ti.kernel
def semiEuler():
    # semi-Euler update pos & vel
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] = vel[i] + h * gravity + attractor_strength[None] * (
                attractor_pos[None] - pos[i])
            oldPos[i] = pos[i]
            pos[i] = pos[i] + h * vel[i]

@ti.kernel
def updteVelocity():
    # update velocity
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def computeGradientVector():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        sumInvMass = invM0 + invM1 + invM2
        if sumInvMass < 1.0e-6:
            print("wrong invMass function")
        D_s = ti.Matrix.cols([b - a, c - a])
        F[i] = D_s @ B[i]
        U, S, V = ti.svd(F[i])
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        g0, g1, g2, isSuccess = computeGradient_f(i, U, S, V)
        if isSuccess:
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
            ) + invM2 * g2.norm_sqr()
            dLambda[i] = -(constraint + alpha * lagrangian[i]) / (l + alpha)
            lagrangian[i] = lagrangian[i] + dLambda[i]
            gradient[3 * i + 0]  = g0
            gradient[3 * i + 1]  = g1
            gradient[3 * i + 2]  = g2

@ti.kernel
def updatePos():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        if (invM0 != 0.0):
            pos[ia] += omega * invM0 * dLambda[i] * gradient[3 * i + 0]
        if (invM1 != 0.0):
            pos[ib] += omega * invM1 * dLambda[i] * gradient[3 * i + 1]
        if (invM2 != 0.0):
            pos[ic] += omega * invM2 * dLambda[i] * gradient[3 * i + 2]


if __name__ == "__main__":
    # pos.from_numpy(0.2 * vertices[:,0:3:2]) # for .obj file
    pos.from_numpy(vertices)  # for txt file
    f2v.from_numpy(triangles)
    init_pos()
    pause = False
    gui = ti.GUI('Corotated FEM XPBD')
    programStart = time.time()
    sumDrawTime = 0
    drawPoint = True
    while gui.running:
        realStart = time.time()
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                pause = not pause
            elif e.key == 'p':
                drawPoint = not drawPoint
        mouse_pos = gui.get_cursor_pos()
        attractor_pos[None] = mouse_pos
        attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(
            gui.RMB)
        gui.circle(mouse_pos, radius=15, color=0x336699)
        if not pause:
            for i in range(NumSteps):
                semiEuler()
                resetLagrangian()
                for ite in range(MaxIte):
                    computeGradientVector()
                    updatePos()
                updteVelocity()
        ti.sync()
        start = time.time()
        if not drawPoint:
            faces = f2v.to_numpy()
            for i in range(NF):
                ia, ib, ic = faces[i]
                a, b, c = pos[ia], pos[ib], pos[ic]
                gui.triangle(a, b, c, color=0x00FF00)

        positions = pos.to_numpy()
        if drawPoint:
            gui.circles(positions, radius=4, color=0xFF00)
        for i in range(staticPoint):
            gui.circle(positions[i], radius=5, color=0xFF0000)
        gui.show()
        end = time.time()
        sumDrawTime += end - start
    programEnd = time.time()
    print("Average Draw Time Ratio: ",
          sumDrawTime / (programEnd - programStart))