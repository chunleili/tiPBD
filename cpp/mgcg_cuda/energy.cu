#include <cuda.h>
#include <cuda_runtime.h>
#include<Eigen/Dense>
#include<iostream>
#include "cuda_utils.cuh"

using real = float;
using Mat3 = Eigen::Matrix<real, 3, 3>;
using Vec3 = Eigen::Matrix<real, 3, 1>;

__device__ real Determinant(const Mat3& A) {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  }
  
  __device__ Vec3 atomicAdd(Vec3& target, const Vec3& val) {
    Vec3 ret;
    ret[0] = ::atomicAdd(&target.data()[0], val[0]);
    ret[1] = ::atomicAdd(&target.data()[1], val[1]);
    ret[2] = ::atomicAdd(&target.data()[2], val[2]);
    return ret;
  }

__device__ void GetRotation(const Mat3& F, Mat3& R) {
    Mat3 C = F.transpose() * F;
    Mat3 C2 = C * C;
    real det = Determinant(F);
    real I_C = C(0, 0) + C(1, 1) + C(2, 2);
    real I_C2 = I_C * I_C;
    real II_C = real(0.5) * (I_C2 - C2(0, 0) - C2(1, 1) - C2(2, 2));
    real III_C = det * det;
    real k = I_C2 - 3 * II_C;
  
    Mat3 U_inv = Mat3::Zero();
    if (k < real(1e-7)) {
      real lambda_inv = 1.0 / sqrt(I_C / 3);
      U_inv(0, 0) = lambda_inv;
      U_inv(1, 1) = lambda_inv;
      U_inv(2, 2) = lambda_inv;
    } else {
      real l = I_C * (I_C2 - real(4.5) * II_C) + real(13.5) * III_C;
      real k_root = sqrt(k);
      real value = l / (k * k_root);
      if (value < -1.0) value = -1.0;
      if (value > 1.0) value = 1.0;
      real phi = acos(value);
      real lambda2 = (I_C + 2 * k_root * cos(phi / 3)) / 3;
      real lambda = sqrt(lambda2);
  
      real III_U = sqrt(III_C);
      if (det < 0) III_U = -III_U;
      real I_U = lambda + sqrt(-lambda2 + I_C + 2 * III_U / lambda);
      real II_U = (I_U * I_U - I_C) / 2;
  
      real inv_rate = 1 / (I_U * II_U - III_U);
      real factor = I_U * III_U * inv_rate;
      Mat3 U = factor * Mat3::Identity();
      factor = (I_U * I_U - II_U) * inv_rate;
      U += factor * C - inv_rate * C2;
  
      inv_rate = 1 / III_U;
      factor = II_U * inv_rate;
      U_inv(0, 0) = factor;
      U_inv(1, 1) = factor;
      U_inv(2, 2) = factor;
      factor = -I_U * inv_rate;
      U_inv += factor * U + inv_rate * C;
    }
  
    R = F * U_inv;
  }


__global__ void EnergyPD(const Vec3 *X, const uint32_t *tet, const Mat3 *Dm_inv,
                         const real *vol, const real mu, real *out,
                         const uint32_t n_tet)
{
    uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_tet)
        return;

    const uint32_t &v1 = tet[4 * t];
    const uint32_t &v2 = tet[4 * t + 1];
    const uint32_t &v3 = tet[4 * t + 2];
    const uint32_t &v4 = tet[4 * t + 3];
    Mat3 Ds;
    Ds.col(0) = X[v1] - X[v4];
    Ds.col(1) = X[v2] - X[v4];
    Ds.col(2) = X[v3] - X[v4];
    Mat3 F = Ds * Dm_inv[t];
    Mat3 R;
    GetRotation(F, R);

    real e = 0;
#pragma unroll
    for (int i = 0; i < 9; ++i)
        e += (F.data()[i] - R.data()[i]) * (F.data()[i] - R.data()[i]);
    e = e * mu * vol[t];

    ::atomicAdd(out, e);
}




#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


/**
 * @brief Computes the energy of a deformable object using CUDA.
 * 
 * This function launches a CUDA kernel to compute the energy of a deformable
 * object based on its vertices, tetrahedral elements, and material properties.
 * 
 * @param X Pointer to an array of Vec3 structures representing the vertices of the object.
 * @param n_verts_ Number of vertices in the object.
 * @param indices_ Pointer to an array of uint32_t representing the indices of the tetrahedral elements.
 * @param n_tet_ Number of tetrahedral elements in the object.
 * @param Dm_inv_ Pointer to an array of Mat3 structures representing the inverse of the deformation gradient matrices.
 * @param volumes_ Pointer to an array of real values representing the volumes of the tetrahedral elements.
 * @param mu Shear modulus of the material.
 * @param E Pointer to an array of real values where the computed energy will be stored.
 */
extern "C" DLLEXPORT 
void compute_energy(const Vec3* X, const uint32_t n_verts_,
     const uint32_t* indices_,const uint32_t n_tet_,
     const Mat3 *Dm_inv_,
     const real *volumes_,
     const real mu,
     real *E )
{
    int threads_per_block_ = 64;
    int tet_threads_per_block_ = 64;
    int tet_blocks_ =
    (n_tet_ + tet_threads_per_block_ - 1) / tet_threads_per_block_;

    real* dE;
    CHECK_CUDA(cudaMalloc(&dE, sizeof(real)));
    CHECK_CUDA(cudaMemset(dE, 0, sizeof(real)));

    Vec3* dX;
    CHECK_CUDA(cudaMalloc(&dX, n_verts_ * sizeof(Vec3)));
    CHECK_CUDA(cudaMemcpy(dX, X, n_verts_ * sizeof(Vec3), cudaMemcpyHostToDevice));

    uint32_t* dindices_;
    CHECK_CUDA(cudaMalloc(&dindices_, n_tet_ * 4 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(dindices_, indices_, n_tet_ * 4 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    Mat3* dDm_inv_;
    CHECK_CUDA(cudaMalloc(&dDm_inv_, n_tet_ * sizeof(Mat3)));
    CHECK_CUDA(cudaMemcpy(dDm_inv_, Dm_inv_, n_tet_ * sizeof(Mat3),
                          cudaMemcpyHostToDevice));

    real* dvolumes_;
    CHECK_CUDA(cudaMalloc(&dvolumes_, n_tet_ * sizeof(real)));
    CHECK_CUDA(cudaMemcpy(dvolumes_, volumes_, n_tet_ * sizeof(real),
                          cudaMemcpyHostToDevice));


    EnergyPD<<<tet_blocks_, threads_per_block_>>>(
        X, dindices_, dDm_inv_, dvolumes_, mu, dE, n_tet_);
    

    CHECK_CUDA(cudaMemcpy(&E, dE, sizeof(real), cudaMemcpyDeviceToHost));
    std::cout<<"Energy: "<<E<<std::endl;
}