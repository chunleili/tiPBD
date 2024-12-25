#pragma once


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>


#include "Vec.h"
#include "CSR.h"


namespace fastmg {

extern bool verbose; // FIXME: This is global
    
struct Buffer {
    void *m_data;
    size_t m_cap;

    Buffer() noexcept;
    Buffer(Buffer &&that) noexcept;
    Buffer &operator=(Buffer &&that) noexcept;
    ~Buffer() noexcept;
    void reserve(size_t new_cap);

    size_t capacity() const noexcept {
        return m_cap;
    }

    void const *data() const noexcept {
        return m_data;
    }
    
    void *data() noexcept {
        return m_data;
    }
};



struct CusparseWrappers {
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    cusolverSpHandle_t cusolverH;
    Buffer buff;

    CusparseWrappers();
    CusparseWrappers(CusparseWrappers &&) = delete;
    ~CusparseWrappers();

    // out = alpha * A@x + beta * out
    void spmv(Vec<float> &out, float const &alpha, CSR<float> const &A, Vec<float> const &x, float const &beta, Buffer &buffer);
    void spgemm(CSR<float> const &matA_,  CSR<float> const &matB_, CSR<float> &matC_);
    // dst = src + alpha * dst
    void axpy(Vec<float> &dst, float const &alpha, Vec<float> const &src);
    void zero(Vec<float> &dst);
    void copy(Vec<float> &dst, Vec<float> const &src);
    // dst = alpha * x
    void scal2(Vec<float> &dst, float const &alpha, Vec<float> const &x);
    // dst = alpha * dst
    void scal(Vec<float> &dst, float const &alpha);
    float vdot(Vec<float> const &x, Vec<float> const &y);
    float vnorm(Vec<float> const &x);

    // x = A^{-1} b by cusolver cholesky
    // https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csrlsvchol
    void spsolve(Vec<float> &x, CSR<float> const &A, Vec<float> &b);

    // transpose csr matrix A to AT
    // https://docs.nvidia.com/cuda/cusparse/index.html?highlight=cusparseCsr2cscEx2#cusparsecsr2cscex2
    void transpose(CSR<float> const & A, CSR<float>& AT);

    //Calculate the largest eigenvalue of a symmetric matrix using the power method!
    // https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csreigvsi  (cusolverSpScsreigvsi is not used here, but it is another option, so I just keep the note. It use the shift inverse method to solve this equation Ax=lam x)
    // Reference code: https://github.com/physicslog/maxEigenValueGPU/blob/25e0aa3d6c9bbeb03be6249d0ab8cfaafd32188c/maxeigenvaluepower.cu#L255
    float computeMaxEigenvaluePowerMethodOptimized(CSR<float>& M, int max_iter);
};

} // namespace fastmg