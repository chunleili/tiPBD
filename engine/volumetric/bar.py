import taichi as ti
import numpy as np

import sys,os
sys.path.append(os.getcwd())

from engine.metadata import meta

meta.youngs_modulus = 1.0
meta.poissons_ratio = 0.3
meta.lame_mu = meta.youngs_modulus / 2 / (1+meta.poissons_ratio)
meta.lame_lambda = meta.youngs_modulus * meta.poissons_ratio / (1+meta.poissons_ratio) / (1-2*meta.poissons_ratio)

def computeGreenStrainAndPiolaStress(x1,x2,x3,x4,invRestMat,restVolume,mu_,lambda_,epsilon,sigma,energy):
    # Determine \partial m/\partial x_i
    F = np.zeros((3,3))
    p14 = x1 - x4
    p24 = x2 - x4
    p34 = x3 - x4
    F[0,0] = p14[0]*invRestMat[0,0] + p24[0]*invRestMat[1,0] + p34[0]*invRestMat[2,0]
    F[0,1] = p14[0]*invRestMat[0,1] + p24[0]*invRestMat[1,1] + p34[0]*invRestMat[2,1]
    F[0,2] = p14[0]*invRestMat[0,2] + p24[0]*invRestMat[1,2] + p34[0]*invRestMat[2,2]

    F[1,0] = p14[1]*invRestMat[0,0] + p24[1]*invRestMat[1,0] + p34[1]*invRestMat[2,0]
    F[1,1] = p14[1]*invRestMat[0,1] + p24[1]*invRestMat[1,1] + p34[1]*invRestMat[2,1]
    F[1,2] = p14[1]*invRestMat[0,2] + p24[1]*invRestMat[1,2] + p34[1]*invRestMat[2,2]

    F[2,0] = p14[2]*invRestMat[0,0] + p24[2]*invRestMat[1,0] + p34[2]*invRestMat[2,0]
    F[2,1] = p14[2]*invRestMat[0,1] + p24[2]*invRestMat[1,1] + p34[2]*invRestMat[2,1]
    F[2,2] = p14[2]*invRestMat[0,2] + p24[2]*invRestMat[1,2] + p34[2]*invRestMat[2,2]


    # epsilon = 1/2 F^T F - I
    epsilon[0,0] = 0.5*(F[0,0]*F[0,0] + F[1,0]*F[1,0] + F[2,0]*F[2,0] - 1.0) # xx
    epsilon[1,1] = 0.5*(F[0,1]*F[0,1] + F[1,1]*F[1,1] + F[2,1]*F[2,1] - 1.0) # yy
    epsilon[2,2] = 0.5*(F[0,2]*F[0,2] + F[1,2]*F[1,2] + F[2,2]*F[2,2] - 1.0) # zz
    epsilon[0,1] = 0.5*(F[0,0]*F[0,1] + F[1,0]*F[1,1] + F[2,0]*F[2,1]) # xy
    epsilon[0,2] = 0.5*(F[0,0]*F[0,2] + F[1,0]*F[1,2] + F[2,0]*F[2,2]) # xz
    epsilon[1,2] = 0.5*(F[0,1]*F[0,2] + F[1,1]*F[1,2] + F[2,1]*F[2,2]) # yz
    epsilon[1,0] = epsilon[0,1]
    epsilon[2,0] = epsilon[0,2]
    epsilon[2,1] = epsilon[1,2]

    # P(F) = F(2 mu E + lambda tr(E)I) => E = green strain
    trace = epsilon[0,0] + epsilon[1,1] + epsilon[2,2]
    ltrace = lambda_*trace
    sigma = epsilon * 2.0*mu_    
    sigma[0,0] += ltrace
    sigma[1,1] += ltrace
    sigma[2,2] += ltrace
    sigma = F @ sigma

    psi = 0.0
    for j in range(3):
        for k in range(3):
            psi += epsilon[j,k] * epsilon[j,k]
    psi = mu_*psi + 0.5*lambda_*trace*trace
    energy = restVolume * psi
    ...


def solve_FEMTetraConstraint(p0, invMass0, p1, invMass1, p2, invMass2, p3, invMass3, restVolume, invRestMat, youngsModulus, poissonRatio, handleInversion, corr0, corr1, corr2, corr3):

    C = 0.0
    gradC = np.zeros((4,3),float)
    epsilon = np.zeros((3,3),float)
    sigma = np.zeros((3,3),float)
    # volume = (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0
    volume = np.cross(p1 - p0, p2 - p0)
    volume = np.dot(volume, p3 - p0) / 6.0

    mu = youngsModulus / 2.0 / (1.0 + poissonRatio)
    lambda_ = youngsModulus * poissonRatio / (1.0 + poissonRatio) / (1.0 - 2.0 * poissonRatio)

    if not handleInversion or volume > 0.0:
        computeGreenStrainAndPiolaStress(p0, p1, p2, p3, invRestMat, restVolume, mu, lambda_, epsilon, sigma, C)
        computeGradCGreen(restVolume, invRestMat, sigma, gradC)

    sum_normGradC = invMass0 * np.linalg.norm(gradC[0])**2 + invMass1 * np.linalg.norm(gradC[1])**2 + invMass2 * np.linalg.norm(gradC[2])**2 + invMass3 * np.linalg.norm(gradC[3])**2

    eps = 1e-6
    if sum_normGradC < eps:
        return False

    # compute scaling factor
    s = C / sum_normGradC

    corr0 = -s * invMass0 * gradC[0]
    corr1 = -s * invMass1 * gradC[1]
    corr2 = -s * invMass2 * gradC[2]
    corr3 = -s * invMass3 * gradC[3]

    if invMass0 > 0.0:
        x1 += corr0
    if invMass1 > 0.0:
        x2 += corr1
    if invMass2 > 0.0:
        x3 += corr2
    if invMass3 > 0.0:
        x4 += corr3

    return True


def computeGradCGreen(restVolume, invRestMat, sigma, J):
    H = np.zeros((3,3),float)
    T = np.zeros((3,3),float)
    T = invRestMat.transpose()
    H = sigma @ T * restVolume

    J[0][0] = H[0,0]
    J[1][0] = H[0,1]
    J[2][0] = H[0,2]

    J[0][1] = H[1,0]
    J[1][1] = H[1,1]
    J[2][1] = H[1,2]

    J[0][2] = H[2,0]
    J[1][2] = H[2,1]
    J[2][2] = H[2,2]

    J[3] = -J[0] - J[1] - J[2]
    

# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #

def test_computeGreenStrainAndPiolaStress():
    x1 = np.array([0, -0.75, -0.75])
    x2 = np.array([0.34482758620689657, -0.75, -0.375])
    x3 = np.array([0.0, -0.375, -0.375])
    x4 = np.array([0.0, -0.75, -0.375])
    
    invRestMat = np.zeros(9,float)
    invRestMat[0]=   0
    invRestMat[1]=    2.8999999999999995
    invRestMat[2]=    0
    invRestMat[3]=    0
    invRestMat[4]=    0
    invRestMat[5]=    2.6666666666666665
    invRestMat[6]=    -2.6666666666666665
    invRestMat[7]=    0
    invRestMat[8]=    0
    invRestMat = invRestMat.reshape((3,3)).T
    restVolume = 0.0080818965517241385
    mu = 0.38461538461538458
    lambda_ = 0.57692307692307687
    epsilon = np.empty((3,3),float)
    sigma = np.empty((3,3),float)
    energy = 0.0
    computeGreenStrainAndPiolaStress(x1,x2,x3,x4,invRestMat,restVolume,mu,lambda_,epsilon,sigma,energy)



def test_solve_FEMTetraConstraint():
    invMass0 = 0
    invMass1 = 1.0
    invMass2 = 1.0
    invMass3 = 0

    x1 = np.array([0, -0.75, -0.75])
    x2 = np.array([
        0.34482758620689657,
        -0.76977695999999796,
        -0.37500000000000000])
    x3 = np.array([
        0.34482758620689657,
        -0.39477695999999796,
        -0.75000000000000000])
    x4 = np.array([0.0, -0.375, -0.375])

    restVolume = 0.016163793103448277

    invRestMat = np.zeros(9,float)

    invRestMat[0]=-1.4499999999999997
    invRestMat[1]=1.4499999999999997
    invRestMat[2]=1.4499999999999997
    invRestMat[3]=-1.3333333333333333
    invRestMat[4]=-1.3333333333333333
    invRestMat[5]=1.3333333333333333
    invRestMat[6]=-1.3333333333333333
    invRestMat[7]=1.3333333333333333
    invRestMat[8]=-1.3333333333333333
    invRestMat = invRestMat.reshape((3,3)).T

    youngsModulus = 1.0
    poissonRatio = 0.3
    handleInversion = False

    corr1 = np.zeros(3,float)
    corr2 = np.zeros(3,float)
    corr3 = np.zeros(3,float)
    corr4 = np.zeros(3,float)

    solve_FEMTetraConstraint(x1, invMass0, x2, invMass1, x3, invMass2, x4, invMass3, restVolume, invRestMat, youngsModulus, poissonRatio, handleInversion, corr1, corr2, corr3, corr4)




if __name__ == "__main__":
    test_solve_FEMTetraConstraint()