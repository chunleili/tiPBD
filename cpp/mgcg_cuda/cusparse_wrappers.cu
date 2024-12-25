
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
#include "cusparse_wrappers.h"

namespace fastmg{


extern bool verbose=false; // FIXME: This is global


template <class T>
cudaDataType_t cudaDataTypeFor();

template <>
cudaDataType_t cudaDataTypeFor<int8_t>() {
    return CUDA_R_8I;
}

template <>
cudaDataType_t cudaDataTypeFor<uint8_t>() {
    return CUDA_R_8U;
}

template <>
cudaDataType_t cudaDataTypeFor<int16_t>() {
    return CUDA_R_16I;
}

template <>
cudaDataType_t cudaDataTypeFor<uint16_t>() {
    return CUDA_R_16U;
}

template <>
cudaDataType_t cudaDataTypeFor<int32_t>() {
    return CUDA_R_32I;
}

template <>
cudaDataType_t cudaDataTypeFor<uint32_t>() {
    return CUDA_R_32U;
}

template <>
cudaDataType_t cudaDataTypeFor<int64_t>() {
    return CUDA_R_64I;
}

template <>
cudaDataType_t cudaDataTypeFor<uint64_t>() {
    return CUDA_R_64U;
}

template <>
cudaDataType_t cudaDataTypeFor<nv_half>() {
    return CUDA_R_16F;
}

template <>
cudaDataType_t cudaDataTypeFor<nv_bfloat16>() {
    return CUDA_R_16BF;
}

template <>
cudaDataType_t cudaDataTypeFor<float>() {
    return CUDA_R_32F;
}

template <>
cudaDataType_t cudaDataTypeFor<double>() {
    return CUDA_R_64F;
}



struct DnVec {
    cusparseDnVecDescr_t handle;

    operator cusparseDnVecDescr_t() const noexcept {
        return handle;
    }

    DnVec() noexcept : handle(0) {}

    template <class T>
    DnVec(Vec<T> &v) {
        CHECK_CUSPARSE(cusparseCreateDnVec(&handle, v.size(), v.data(), cudaDataTypeFor<T>()));
    }

    DnVec(DnVec &&that) noexcept : handle(that.handle) {
        that.handle = nullptr;
    }

    DnVec &operator=(DnVec &&that) noexcept {
        if (this == &that) return *this;
        if (handle)
            CHECK_CUSPARSE(cusparseDestroyDnVec(handle));
        handle = that.handle;
        that.handle = nullptr;
        return *this;
    }

    ~DnVec() {
        if (handle)
            CHECK_CUSPARSE(cusparseDestroyDnVec(handle));
    }
};

struct ConstDnVec {
    cusparseConstDnVecDescr_t handle;

    operator cusparseConstDnVecDescr_t() const noexcept {
        return handle;
    }

    ConstDnVec() noexcept : handle(0) {}

    template <class T>
    ConstDnVec(Vec<T> const &v) {
        CHECK_CUSPARSE(cusparseCreateConstDnVec(&handle, v.size(), v.data(), cudaDataTypeFor<T>()));
    }

    ConstDnVec(ConstDnVec &&that) noexcept : handle(that.handle) {
        that.handle = nullptr;
    }

    ConstDnVec &operator=(ConstDnVec &&that) noexcept {
        if (this == &that) return *this;
        if (handle)
            CHECK_CUSPARSE(cusparseDestroyDnVec(handle));
        handle = that.handle;
        that.handle = nullptr;
        return *this;
    }

    ~ConstDnVec() {
        if (handle)
            CHECK_CUSPARSE(cusparseDestroyDnVec(handle));
    }
};




// container of handle and descriptor
struct SpMat {
    cusparseSpMatDescr_t handle;

    operator cusparseSpMatDescr_t() const noexcept {
        return handle;
    }

    SpMat() noexcept : handle(0) {}

    template <class T>
    SpMat(CSR<T> &m) {
        CHECK_CUSPARSE(cusparseCreateCsr(&handle, m.nrows, m.ncols, m.numnonz,
                                         m.indptr.data(), m.indices.data(), m.data.data(),
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, cudaDataTypeFor<T>()) );
    }

    SpMat(SpMat &&that) noexcept : handle(that.handle) {
        that.handle = nullptr;
    }

    SpMat &operator=(SpMat &&that) noexcept {
        if (this == &that) return *this;
        if (handle)
            CHECK_CUSPARSE(cusparseDestroySpMat(handle));
        handle = that.handle;
        that.handle = nullptr;
        return *this;
    }

    ~SpMat() {
        if (handle)
            CHECK_CUSPARSE(cusparseDestroySpMat(handle));
    }
};

// container of handle and descriptor, const version
struct ConstSpMat {
    cusparseConstSpMatDescr_t handle;

    operator cusparseConstSpMatDescr_t() const noexcept {
        return handle;
    }

    ConstSpMat() noexcept : handle(0) {}

    template <class T>
    ConstSpMat(CSR<T> const &m) {
        CHECK_CUSPARSE(cusparseCreateConstCsr(&handle, m.nrows, m.ncols, m.numnonz,
                                              m.indptr.data(), m.indices.data(), m.data.data(),
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, cudaDataTypeFor<T>()) );
    }

    ConstSpMat(SpMat &&that) noexcept : handle(that.handle) {
        that.handle = nullptr;
    }

    ConstSpMat &operator=(ConstSpMat &&that) noexcept {
        if (this == &that) return *this;
        if (handle)
            CHECK_CUSPARSE(cusparseDestroySpMat(handle));
        handle = that.handle;
        that.handle = nullptr;
        return *this;
    }

    ~ConstSpMat() {
        if (handle)
            CHECK_CUSPARSE(cusparseDestroySpMat(handle));
    }
};



    Buffer::Buffer() noexcept : m_data(nullptr), m_cap(0) {
    }

    Buffer::Buffer(Buffer &&that) noexcept : m_data(that.m_data), m_cap(that.m_cap) {
        that.m_data = nullptr;
        that.m_cap = 0;
    }

    Buffer& Buffer::operator=(Buffer &&that) noexcept {
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

    Buffer::~Buffer() noexcept {
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
    }

    void Buffer::reserve(size_t new_cap) {
        if (m_cap < new_cap) {
            if (m_data)
                CHECK_CUDA(cudaFree(m_data));
            m_data = nullptr;
            CHECK_CUDA(cudaMalloc(&m_data, new_cap));
            m_cap = new_cap;
        }
    }



/* -------------------------------------------------------------------------- */
/*                              cusparse wrappers                             */
/* -------------------------------------------------------------------------- */


// out = alpha * A@x + beta * out
void CusparseWrappers::spmv(Vec<float> &out, float const &alpha, CSR<float> const &A, Vec<float> const &x, float const &beta, Buffer &buffer) {
    assert(out.size() == A.nrows);
    size_t bufSize = 0;
    ConstSpMat dA(A);
    ConstDnVec dx(x);
    DnVec dout(out);
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, dA, dx, &beta,
                                            dout, cudaDataTypeFor<float>(),
                                            CUSPARSE_SPMV_ALG_DEFAULT, &bufSize));
    buffer.reserve(bufSize);
    CHECK_CUSPARSE(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, dA, dx, &beta,
                                dout, cudaDataTypeFor<float>(),
                                CUSPARSE_SPMV_ALG_DEFAULT, buffer.data()));
}

// C = A * B
void CusparseWrappers::spgemm(CSR<float> const &matA_,  CSR<float> const &matB_, CSR<float> &matC_) 
{
    ConstSpMat descA(matA_); //descriptor for A
    ConstSpMat descB(matB_);
    matC_.resize(matA_.nrows, matB_.ncols, 0);
    SpMat descC(matC_);
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/ade391a17672d26e55429035450bc44afd277d34/cuSPARSE/spgemm/spgemm_example.c#L161
    // https://docs.nvidia.com/cuda/cusparse/#cusparsespgemm
    //--------------------------------------------------------------------------
    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // buffers
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    Buffer dBuffer1, dBuffer2;
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(cusparse, opA, opB,
                                    &alpha, descA, descB, &beta, descC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, NULL) )
    // CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    dBuffer1.reserve(bufferSize1);

    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(cusparse, opA, opB,
                                    &alpha, descA, descB, &beta, descC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1.data()) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(cusparse, opA, opB,
                            &alpha, descA, descB, &beta, descC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, NULL) )
    dBuffer2.reserve(bufferSize2);

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(cusparse, opA, opB,
                                        &alpha, descA, descB, &beta, descC,
                                        computeType, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize2, dBuffer2.data()) )
    // --------------------------------------------------------------------------
    // get matrix C non-zero entries C_nnz1
    CHECK_CUSPARSE( cusparseSpMatGetSize(descC, &matC_.nrows, &matC_.ncols, &matC_.numnonz) )
    // allocate matrix C
    matC_.resize(matC_.nrows, matC_.ncols, matC_.numnonz);
    // update matC with the new pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(descC, matC_.indptr.data(), matC_.indices.data(), matC_.data.data()) )

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(cusparse, opA, opB,
                            &alpha, descA, descB, &beta, descC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )
}



// transpose csr matrix A to AT
// https://docs.nvidia.com/cuda/cusparse/index.html?highlight=cusparseCsr2cscEx2#cusparsecsr2cscex2
void CusparseWrappers::transpose(CSR<float> const & A, CSR<float>& AT)
{
    int m = A.nrows;
    int n = A.ncols;
    int nnz = A.numnonz;
    const float *csrVal  = A.data.data();
    const int *csrRowPtr = A.indptr.data();
    const int *csrColInd = A.indices.data();
    float *cscVal  = AT.data.data();
    int *cscColPtr = AT.indptr.data();
    int *cscRowInd = AT.indices.data();
    cudaDataType  valType = CUDA_R_32F;
    cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cusparseCsr2CscAlg_t    alg = CUSPARSE_CSR2CSC_ALG_DEFAULT;
    cusparseStatus_t status;
    size_t bufferSize = 0;
    Buffer buffer;

    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(cusparse, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, &bufferSize));
    buffer.reserve(bufferSize);
    CHECK_CUSPARSE( cusparseCsr2cscEx2(           cusparse, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer.data()));                
}



// Generate random number in the range [0, 1)
struct genRandomNumber {
    __host__ __device__
    float operator()(const int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }
};


//Calculate the largest eigenvalue of a symmetric matrix using the power method!
// https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csreigvsi  (cusolverSpScsreigvsi is not used here, but it is another option, so I just keep the note. It use the shift inverse method to solve this equation Ax=lam x)
// Reference code: https://github.com/physicslog/maxEigenValueGPU/blob/25e0aa3d6c9bbeb03be6249d0ab8cfaafd32188c/maxeigenvaluepower.cu#L255
float CusparseWrappers::computeMaxEigenvaluePowerMethodOptimized(CSR<float>& M, int max_iter) {
    // // Terminal output color (just for cosmetic purpose)
    // #define RST  "\x1B[37m"  // Reset color to white
    // #define KGRN  "\033[0;32m"   // Define green color
    // #define RD "\x1B[31m"  // Define red color
    // #define FGRN(x) KGRN x RST  // Define compiler function for green color
    // #define FRD(x) RD x RST  // Define compiler function for red color

  assert(M.nrows == M.ncols);

  // Initialize two vectors x_i and x_k
  thrust::device_vector<float> x_i(M.nrows), x_k(M.nrows, 0.0f);

  // Set x_i := the random vector
    thrust::transform(thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(M.nrows),
    x_i.begin(),
    genRandomNumber());

  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matM;
  cusparseDnVecDescr_t xi, xk;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  CHECK_CUSPARSE( cusparseCreate(&handle) )

  CHECK_CUSPARSE( cusparseCreateCsr(&matM, M.nrows, M.ncols, M.numnonz,
                                   thrust::raw_pointer_cast(M.indptr.data()),
                                   thrust::raw_pointer_cast(M.indices.data()),
                                   thrust::raw_pointer_cast(M.data.data()),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

  CHECK_CUSPARSE( cusparseCreateDnVec(&xi, M.nrows, thrust::raw_pointer_cast(x_i.data()), CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateDnVec(&xk, M.nrows, thrust::raw_pointer_cast(x_k.data()), CUDA_R_32F) )

  CHECK_CUSPARSE( cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )

  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  float max_eigenvalue(0.0f), max_eigenvalue_prev(0.0f);
  float tol = 1e-4;  // tolerance for convergence
  int itr = 0;
  float err = 0.0f;
  // Power iteration method
  while (itr < max_iter) {
    // Compute x_k = A * x_i; generates Krylov subspace
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )

    // Compute the L2 norm of x_k
    float norm = std::sqrt(thrust::inner_product(x_k.begin(), x_k.end(), x_k.begin(), 0.0f));

    // Normalize x_k and update x_i
    thrust::transform(x_k.begin(), x_k.end(), x_i.begin(), thrust::placeholders::_1 / norm);

    // Compute the maximum eigenvalue
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )

    max_eigenvalue = thrust::inner_product(x_i.begin(), x_i.end(), x_k.begin(), 0.0f);


    err = std::abs(max_eigenvalue - max_eigenvalue_prev);
    if (err < tol && itr >= 10) {
        if (verbose)
        {
            std::cout << ("[NOTE]: ") << "Converged at iterations: " << itr << std::endl;
        }
      return max_eigenvalue;
    }

    max_eigenvalue_prev = max_eigenvalue;
    itr++;
  }

  // Destroy the handle and descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matM) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(xi) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(xk) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  CHECK_CUDA( cudaFree(dBuffer) )

    if (verbose)
    {
        std::cout << ("\n[NOTE]: ") << "Max_iter("<<max_iter<<") reached when calculating max eig, error=" <<err<< std::endl;  // no convergence
    }
  return max_eigenvalue;
}


void CusparseWrappers::spsolve(Vec<float> &x, CSR<float> const &A, Vec<float> &b) {
    cusparseMatDescr_t descrA = NULL;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 
    int singularity;

    CHECK_CUSOLVER( cusolverSpScsrlsvchol(cusolverH, A.nrows, A.numnonz, descrA, A.data.data(), A.indptr.data(), A.indices.data(), b.data(), 1e-10, 0, x.data(), &singularity) );
}




CusparseWrappers::CusparseWrappers() {
    CHECK_CUSPARSE(cusparseCreate(&cusparse));
    CHECK_CUBLAS(cublasCreate_v2(&cublas));
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverH));
}


CusparseWrappers::~CusparseWrappers() {
    CHECK_CUSPARSE(cusparseDestroy(cusparse));
    CHECK_CUBLAS(cublasDestroy_v2(cublas));
    CHECK_CUSOLVER(cusolverSpDestroy(cusolverH));
}



// dst = src + alpha * dst
void CusparseWrappers::axpy(Vec<float> &dst, float const &alpha, Vec<float> const &src) {
    assert(dst.size() == src.size());
    CHECK_CUBLAS(cublasSaxpy_v2(cublas, dst.size(), &alpha, src.data(), 1, dst.data(), 1));
}

void CusparseWrappers::zero(Vec<float> &dst) {
    CHECK_CUDA(cudaMemset(dst.data(), 0, dst.size() * sizeof(float)));
}

void CusparseWrappers::copy(Vec<float> &dst, Vec<float> const &src) {
    dst.resize(src.size());
    CHECK_CUDA(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToDevice));
}

// dst = alpha * x
void CusparseWrappers::scal2(Vec<float> &dst, float const &alpha, Vec<float> const &x) {
    copy(dst, x);
    CHECK_CUBLAS(cublasSscal_v2(cublas, dst.size(), &alpha, dst.data(), 1));
}

// dst = alpha * dst
void CusparseWrappers::scal(Vec<float> &dst, float const &alpha) {
    CHECK_CUBLAS(cublasSscal_v2(cublas, dst.size(), &alpha, dst.data(), 1));
}

float CusparseWrappers::vdot(Vec<float> const &x, Vec<float> const &y) {
    float result;
    CHECK_CUBLAS(cublasSdot_v2(cublas, x.size(), x.data(), 1, y.data(), 1, &result));
    return result;
}

float CusparseWrappers::vnorm(Vec<float> const &x) {
    float result;
    CHECK_CUBLAS(cublasSnrm2_v2(cublas, x.size(), x.data(), 1, &result));
    return result;
}





} // namespace fastmg
