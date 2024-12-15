#pragma once


#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>
#include <numeric>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>

#include "cuda_utils.cuh"
#include "Vec.h"
#include "CSR.h"


namespace fastmg {

extern bool verbose; // FIXME: This is global
    
struct Buffer {
    void *m_data;
    size_t m_cap;

    Buffer() noexcept : m_data(nullptr), m_cap(0) {
    }

    Buffer(Buffer &&that) noexcept : m_data(that.m_data), m_cap(that.m_cap) {
        that.m_data = nullptr;
        that.m_cap = 0;
    }

    Buffer &operator=(Buffer &&that) noexcept {
        if (this == &that) return *this;
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
        m_data = that.m_data;
        m_cap = that.m_cap;
        that.m_data = nullptr;
        that.m_cap = 0;
        return *this;
    }

    ~Buffer() noexcept {
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
    }

    void reserve(size_t new_cap) {
        if (m_cap < new_cap) {
            if (m_data)
                CHECK_CUDA(cudaFree(m_data));
            m_data = nullptr;
            CHECK_CUDA(cudaMalloc(&m_data, new_cap));
            m_cap = new_cap;
        }
    }

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

    CusparseWrappers() {
        CHECK_CUSPARSE(cusparseCreate(&cusparse));
        CHECK_CUBLAS(cublasCreate_v2(&cublas));
        CHECK_CUSOLVER(cusolverSpCreate(&cusolverH));
    }

    CusparseWrappers(CusparseWrappers &&) = delete;

    ~CusparseWrappers() {
        CHECK_CUSPARSE(cusparseDestroy(cusparse));
        CHECK_CUBLAS(cublasDestroy_v2(cublas));
        CHECK_CUSOLVER(cusolverSpDestroy(cusolverH));
    }

    // out = alpha * A@x + beta * out
    void spmv(Vec<float> &out, float const &alpha, CSR<float> const &A, Vec<float> const &x, float const &beta, Buffer &buffer);
    
    void spgemm(CSR<float> const &matA_,  CSR<float> const &matB_, CSR<float> &matC_);

    // dst = src + alpha * dst
    void axpy(Vec<float> &dst, float const &alpha, Vec<float> const &src) {
        assert(dst.size() == src.size());
        CHECK_CUBLAS(cublasSaxpy_v2(cublas, dst.size(), &alpha, src.data(), 1, dst.data(), 1));
    }

    void zero(Vec<float> &dst) {
        CHECK_CUDA(cudaMemset(dst.data(), 0, dst.size() * sizeof(float)));
    }

    void copy(Vec<float> &dst, Vec<float> const &src) {
        dst.resize(src.size());
        CHECK_CUDA(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // dst = alpha * x
    void scal2(Vec<float> &dst, float const &alpha, Vec<float> const &x) {
        copy(dst, x);
        CHECK_CUBLAS(cublasSscal_v2(cublas, dst.size(), &alpha, dst.data(), 1));
    }

    // dst = alpha * dst
    void scal(Vec<float> &dst, float const &alpha) {
        CHECK_CUBLAS(cublasSscal_v2(cublas, dst.size(), &alpha, dst.data(), 1));
    }

    float vdot(Vec<float> const &x, Vec<float> const &y) {
        float result;
        CHECK_CUBLAS(cublasSdot_v2(cublas, x.size(), x.data(), 1, y.data(), 1, &result));
        return result;
    }

    float vnorm(Vec<float> const &x) {
        float result;
        CHECK_CUBLAS(cublasSnrm2_v2(cublas, x.size(), x.data(), 1, &result));
        return result;
    }

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