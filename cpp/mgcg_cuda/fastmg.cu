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

#include "kernels.cuh"
#include "utils.cuh"

using std::cout;
using std::endl;


namespace {

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





template <class T>
struct Vec {
    T *m_data;
    size_t m_size;
    size_t m_cap;

    Vec() noexcept : m_data(nullptr), m_size(0), m_cap(0) {
    }

    Vec(Vec &&that) noexcept : m_data(that.m_data), m_size(that.m_size), m_cap(that.m_cap) {
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_cap = 0;
    }

    Vec &operator=(Vec &&that) noexcept {
        if (this == &that) return *this;
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
        m_data = that.m_data;
        m_size = that.m_size;
        m_cap = that.m_cap;
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_cap = 0;
        return *this;
    }

    void swap(Vec &that) noexcept {
        std::swap(m_data, that.m_data);
        std::swap(m_size, that.m_size);
        std::swap(m_cap, that.m_cap);
    }

    ~Vec() noexcept {
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
    }

    void resize(size_t new_size) {
        bool change = m_cap < new_size;
        if (change) {
            if (m_data)
                CHECK_CUDA(cudaFree(m_data));
            m_data = nullptr;
            CHECK_CUDA(cudaMalloc(&m_data, sizeof(T) * new_size));
            m_cap = new_size;
        }
        if (m_size != new_size || change) {
            m_size = new_size;
        }
    }

    // host to device
    void assign(T const *data, size_t size) {
        resize(size);
        CHECK_CUDA(cudaMemcpy(m_data, data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    // device to host
    void tohost(std::vector<T> &data_host) const{
        data_host.resize(size());
        CHECK_CUDA(cudaMemcpy(data_host.data(), m_data, sizeof(T) * size(), cudaMemcpyDeviceToHost));
    }

    size_t size() const noexcept {
        return m_size;
    }

    T const *data() const noexcept {
        return m_data;
    }

    T *data() noexcept {
        return m_data;
    }
};


template <class T>
struct HostVec {
    T *m_data;
    size_t m_size;
    size_t m_cap;

    HostVec() noexcept : m_data(nullptr), m_size(0), m_cap(0) {
    }

    HostVec(HostVec &&that) noexcept : m_data(that.m_data), m_size(that.m_size), m_cap(that.m_cap) {
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_cap = 0;
    }

    HostVec &operator=(HostVec &&that) noexcept {
        if (this == &that) return *this;
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
        m_data = that.m_data;
        m_size = that.m_size;
        m_cap = that.m_cap;
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_cap = 0;
        return *this;
    }

    void swap(HostVec &that) noexcept {
        std::swap(m_data, that.m_data);
        std::swap(m_size, that.m_size);
        std::swap(m_cap, that.m_cap);
    }

    ~HostVec() noexcept {
        if (m_data)
            CHECK_CUDA(cudaFreeHost(m_data));
        m_data = nullptr;
    }

    void resize(size_t new_size) {
        bool change = m_cap < new_size;
        if (change) {
            if (m_data)
                CHECK_CUDA(cudaFreeHost(m_data));
            m_data = nullptr;
            CHECK_CUDA(cudaMallocHost(&m_data, sizeof(T) * new_size));
            m_cap = new_size;
        }
        if (m_size != new_size || change) {
            m_size = new_size;
        }
    }

    // host to device
    void assign(T const *data, size_t size) {
        resize(size);
        CHECK_CUDA(cudaMemcpy(m_data, data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    // device to host
    void tohost(std::vector<T> &data_host) const{
        data_host.resize(size());
        CHECK_CUDA(cudaMemcpy(data_host.data(), m_data, sizeof(T) * size(), cudaMemcpyDeviceToHost));
    }

    size_t size() const noexcept {
        return m_size;
    }

    T const *data() const noexcept {
        return m_data;
    }

    T *data() noexcept {
        return m_data;
    }
};


template <typename T=float>
std::vector<T> debug_cuda_vec(Vec<T> &v, std::string name) {
    std::vector<T> v_host(v.size());
    v.tohost(v_host);
    cout<<name<<": ";
    int k=0;
    for(auto i:v_host)
    {
        if(k>10)
            break;
        std::cout<<i<<" ";
        k++;
    }
    std::cout<<endl;
    return v_host;
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


// Data of csr matrix
template <class T>
struct CSR {
    Vec<int> indices;
    Vec<T> data;
    Vec<int> indptr;
    int64_t nrows;
    int64_t ncols;
    int64_t numnonz;

    CSR() noexcept : nrows(0), ncols(0), numnonz(0) {}

    void assign(T const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        indices.resize(nind);
        indptr.resize(nptr);
        data.resize(ndat);
        CHECK_CUDA(cudaMemcpy(data.data(), datap, data.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indices.data(), indicesp, indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indptr.data(), indptrp, indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        nrows = rows;
        ncols = cols;
        numnonz = nnz;
    }

    void resize(size_t rows, size_t cols, size_t nnz) {
        nrows = rows;
        ncols = cols;
        numnonz = nnz;
        data.resize(nnz);
        indices.resize(nnz);
        indptr.resize(rows + 1);
    }

    void tohost(std::vector<T> &data_host, std::vector<int> &indices_host, std::vector<int> &indptr_host) const {
        data_host.resize(data.size());
        indices_host.resize(indices.size());
        indptr_host.resize(indptr.size());
        CHECK_CUDA(cudaMemcpy(data_host.data(), data.data(), data.size() * sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices_host.data(), indices.data(), indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr_host.data(), indptr.data(), indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
};


// template <class T>
// struct SuperCSR:CSR<T>
// {
//     Vec<T> ii;
//     Vec<T> jj;

//     SuperCSR() noexcept : nrows(0), ncols(0), numnonz(0) {}

//     void assign(T const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz, T const *iip, T const *jjp) {
//         indices.resize(nind);
//         indptr.resize(nptr);
//         data.resize(ndat);
//         CHECK_CUDA(cudaMemcpy(data.data(), datap, data.size() * sizeof(T), cudaMemcpyHostToDevice));
//         CHECK_CUDA(cudaMemcpy(indices.data(), indicesp, indices.size() * sizeof(int), cudaMemcpyHostToDevice));
//         CHECK_CUDA(cudaMemcpy(indptr.data(), indptrp, indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
//         nrows = rows;
//         ncols = cols;
//         numnonz = nnz;

//         ii.resize(nnz);
//         jj.resize(nnz);
//         CHECK_CUDA(cudaMemcpy(ii.data(), iip, ii.size() * sizeof(T), cudaMemcpyHostToDevice));
//         CHECK_CUDA(cudaMemcpy(jj.data(), jjp, jj.size() * sizeof(T), cudaMemcpyHostToDevice));
//     }

//     void resize(size_t rows, size_t cols, size_t nnz) {
//         nrows = rows;
//         ncols = cols;
//         numnonz = nnz;
//         data.resize(nnz);
//         indices.resize(nnz);
//         indptr.resize(rows + 1);

//         ii.resize(nnz);
//         jj.resize(nnz);
//     }
// };



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

struct Kernels {
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    cusolverSpHandle_t cusolverH;

    Kernels() {
        CHECK_CUSPARSE(cusparseCreate(&cusparse));
        CHECK_CUBLAS(cublasCreate_v2(&cublas));
        CHECK_CUSOLVER(cusolverSpCreate(&cusolverH));
    }

    Kernels(Kernels &&) = delete;

    ~Kernels() {
        CHECK_CUSPARSE(cusparseDestroy(cusparse));
        CHECK_CUBLAS(cublasDestroy_v2(cublas));
        CHECK_CUSOLVER(cusolverSpDestroy(cusolverH));
    }

    // out = alpha * A@x + beta * out
    void spmv(Vec<float> &out, float const &alpha, CSR<float> const &A, Vec<float> const &x, float const &beta, Buffer &buffer) {
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
    void spgemm(CSR<float> const &matA_,  CSR<float> const &matB_, CSR<float> &matC_) 
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

    // dst = alpha * alpha
    void scal(Vec<float> &dst, float const &alpha) {
        CHECK_CUBLAS(cublasSscal_v2(cublas, dst.size(), &alpha, dst.data(), 1));
    }

    // x = A^{-1} b by cusolver cholesky
    // https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csrlsvchol
    void spsolve(Vec<float> &x, CSR<float> const &A, Vec<float> &b) {
        cusparseMatDescr_t descrA = NULL;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
        CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 
        int singularity;

        CHECK_CUSOLVER( cusolverSpScsrlsvchol(cusolverH, A.nrows, A.numnonz, descrA, A.data.data(), A.indptr.data(), A.indices.data(), b.data(), 1e-10, 0, x.data(), &singularity) );
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

    // transpose csr matrix A to AT
    // https://docs.nvidia.com/cuda/cusparse/index.html?highlight=cusparseCsr2cscEx2#cusparsecsr2cscex2
    void transpose(CSR<float> const & A, CSR<float>& AT)
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



//Calculate the largest eigenvalue of a symmetric matrix using the power method!
// https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csreigvsi  (cusolverSpScsreigvsi is not used here, but it is another option, so I just keep the note. It use the shift inverse method to solve this equation Ax=lam x)
// Reference code: https://github.com/physicslog/maxEigenValueGPU/blob/25e0aa3d6c9bbeb03be6249d0ab8cfaafd32188c/maxeigenvaluepower.cu#L255
float computeMaxEigenvaluePowerMethodOptimized(CSR<float>& M, int max_iter) {
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
  float tol = 1e-3;  // tolerance for convergence
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
    if (err < tol) {
      std::cout << ("[NOTE]: ") << "Converged at iterations: " << itr << std::endl;
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

  std::cout << ("\n[NOTE]: ") << "Max_iter("<<max_iter<<") reached when calculating max eig, error=" <<err<< std::endl;  // no convergence
  return max_eigenvalue;
}
};

struct MGLevel {
    CSR<float> A;
    CSR<float> R;
    CSR<float> P;
    Vec<float> residual;
    Vec<float> b;
    Vec<float> x;
    Vec<float> h;
    Vec<float> outh;
};


struct FastFillCloth : Kernels {
    CSR<float> A;
    float alpha;
    int NE;
    int NV;
    int num_nonz;
    int nrows, ncols;
    Vec<float> d_inv_mass;
    Vec<int> d_ii, d_jj;
    Vec<int> d_edges;
    Vec<float> d_pos;
    Vec<int> d_adjacent_edge_abc;
    Vec<int> d_num_adjacent_edge;

    void fetch_A(float *data_in, int *indices_in, int *indptr_in) {
        CHECK_CUDA(cudaMemcpy(data_in, A.data.data(), sizeof(float) * A.numnonz, cudaMemcpyDeviceToHost));
    }

    void set_data_v2(int* edges_in, int NE_in, float* inv_mass_in, int NV_in, float* pos_in, float alpha_in)
    {
        NE = NE_in;
        NV = NV_in;
        nrows = NE;
        ncols = NE;

        d_edges.assign(edges_in, NE*2);
        d_inv_mass.assign(inv_mass_in, NV);
        d_pos.assign(pos_in, NV*3);

        alpha = alpha_in;
    }

    void update_pos_v2(float* pos_in)
    {
        d_pos.assign(pos_in, NV*3);
        CHECK_CUDA(cudaMemcpy(pos_in, d_pos.data(), sizeof(float) * NV*3, cudaMemcpyDeviceToHost));
    }


    void init_from_python_cache_v2(
        int *adjacent_edge_in,
        int *num_adjacent_edge_in,
        int *adjacent_edge_abc_in,
        int num_nonz_in,
        float *spmat_data_in,
        int *spmat_indices_in,
        int *spmat_indptr_in,
        int *spmat_ii_in,
        int *spmat_jj_in,
        int NE_in,
        int NV_in)
    {
        NE = NE_in;
        NV = NV_in;
        num_nonz = num_nonz_in;

        printf("Copying A, ii, jj\n");
        A.assign(spmat_data_in, num_nonz, spmat_indices_in, num_nonz, spmat_indptr_in, NE+1, NE, NE, num_nonz);
        d_ii.assign(spmat_ii_in, num_nonz);
        d_jj.assign(spmat_jj_in, num_nonz);
        cout<<"Finish."<<endl;

        printf("Copying adj\n");
        d_num_adjacent_edge.assign(num_adjacent_edge_in, NE);
        d_adjacent_edge_abc.resize(NE*60);
        CHECK_CUDA(cudaMemcpy(d_adjacent_edge_abc.data(), adjacent_edge_abc_in, sizeof(int) * NE * 60, cudaMemcpyHostToDevice));
        cout<<"Finish."<<endl;
    }


    void run(float* pos_in)
    {
        update_pos_v2(pos_in);
        fill_A_CSR_gpu();
    }


    void fill_A_CSR_gpu()
    {
        fill_A_CSR_kernel<<<num_nonz / 256 + 1, 256>>>(A.data.data(),
                                                 A.indptr.data(),
                                                 A.indices.data(),
                                                 d_ii.data(),
                                                 d_jj.data(),
                                                 d_adjacent_edge_abc.data(),
                                                 d_num_adjacent_edge.data(),
                                                 num_nonz,
                                                 d_inv_mass.data(),
                                                 alpha,
                                                 NV,
                                                 NE,
                                                 d_edges.data(),
                                                 d_pos.data());
        cudaDeviceSynchronize();
        launch_check();
    }
}; //FastFill struct


struct VCycle : Kernels {
    std::vector<MGLevel> levels;
    size_t nlvs;
    std::vector<float> chebyshev_coeff;
    size_t smoother_type = 1; //1:chebyshev, 2:jacobi, 3:gauss_seidel
    float jacobi_omega;
    size_t jacobi_niter=100;
    Vec<float> z;
    Vec<float> r;
    Vec<float> outer_x;
    Vec<float> x_new;
    Vec<float> outer_b;
    float save_rho_prev;
    Vec<float> save_p;
    Vec<float> save_q;
    Buffer buff;
    float rtol;
    size_t maxiter;
    std::vector<float> residuals;
    size_t niter; //final number of iterations to break the loop
    float max_eig;

    void setup_smoothers_cuda(int type) {
        cout<<"\nSetting up smoothers..."<<endl;
        if(smoother_type == 1)
        {
            setup_chebyshev_cuda(levels[0].A);
        }
        else if (smoother_type == 2)
        {
            //TODO:setup jacobi
            // setup_jacobi_v2(levels[0].A, 100);
        }
    }


    void setup_chebyshev_cuda(CSR<float> &A) {
        float lower_bound=1.0/30.0;
        float upper_bound=1.1;
        float rho = computeMaxEigenvaluePowerMethodOptimized(A, 100);
        float a = rho * lower_bound;
        float b = rho * upper_bound;
        chebyshev_polynomial_coefficients(a, b);
        
        max_eig = rho;
        cout<<"max eigenvalue: "<<max_eig<<endl;
    }


    void chebyshev_polynomial_coefficients(float a, float b)
    {
        int degree=3;
        const float PI = 3.14159265358979323846;

        if(a >= b || a <= 0)
            assert(false && "Invalid input for Chebyshev polynomial coefficients");

        // Chebyshev roots for the interval [-1,1]
        std::vector<float> std_roots(degree);
        for(int i=0; i<degree; i++)
        {
            std_roots[i] = std::cos(PI * (i + 0.5) / (float)degree);
        }

        // Chebyshev roots for the interval [a,b]
        std::vector<float> scaled_roots(degree);
        for(int i=0; i<degree; i++)
        {
            scaled_roots[i] = 0.5 * (b-a) * (1 + std_roots[i]) + a;
        }

        // Compute monic polynomial coefficients of polynomial with scaled roots
        std::vector<float> scaled_poly(4);
        // np.poly for 3 roots. This will calc the coefficients of the polynomial from roots.
        // i.e., (x - root1) * (x - root2) * (x - root3) = x^3 - (root1 + root2 + root3)x^2 + (root1*root2 + root2*root3 + root3*root1)x - root1*root2*root3
        scaled_poly[0] = 1.0;
        scaled_poly[1] = -(scaled_roots[0] + scaled_roots[1] + scaled_roots[2]);
        scaled_poly[2] = scaled_roots[0]*scaled_roots[1] + scaled_roots[1]*scaled_roots[2] + scaled_roots[2]*scaled_roots[0];
        scaled_poly[3] = -scaled_roots[0]*scaled_roots[1]*scaled_roots[2];

        // Scale coefficients to enforce C(0) = 1.0
        float c0 = scaled_poly[3];
        for(int i=0; i<degree; i++)
        {
            scaled_poly[i] /= c0; 
        }


        chebyshev_coeff.resize(degree);
        //CAUTION:setup_chebyshev has "-" at the end
        for(int i=0; i<degree; i++)
        {
            chebyshev_coeff[i] = -scaled_poly[i];
        }

        cout<<"Chebyshev polynomial coefficients: ";
        for(int i=0; i<degree; i++)
        {
            cout<<chebyshev_coeff[i]<<" ";
        }
        cout<<endl;
    }


    float calc_rnorm(Vec<float> const &b, Vec<float> const &x, CSR<float> const &A) {
        float rnorm = 0.0;
        Vec<float> r;
        r.resize(b.size());
        copy(r, b);
        spmv(r, -1, A, x, 1, buff);
        rnorm = vnorm(r);
        return rnorm;
    }


    void setup(size_t numlvs) {
        if (levels.size() < numlvs) {
            levels.resize(numlvs);
        }
        nlvs = numlvs;
        chebyshev_coeff.clear();
        jacobi_omega = 0.0;
    }


    void set_P(size_t lv, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        levels.at(lv).P.assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    }

    void set_A0(float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        levels.at(0).A.assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    }


    // only update the data of A0
    void update_A0(float const *datap) {
        CHECK_CUDA(cudaMemcpy(levels.at(0).A.data.data(), datap, levels.at(0).A.data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }


    void set_A0_from_fastFill(FastFillCloth *ff) {
        levels.at(0).A.data.swap( (ff->A).data);
        levels.at(0).A.indices.swap( (ff->A).indices);
        levels.at(0).A.indptr.swap((ff->A).indptr);
        levels.at(0).A.numnonz = ( ff->num_nonz);
    }


    void chebyshev(int lv, Vec<float> &x, Vec<float> const &b) {
        copy(levels.at(lv).residual, b);
        spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x
        scal2(levels.at(lv).h, chebyshev_coeff.at(0), levels.at(lv).residual); // h = c0 * residual


        for (int i = 1; i < chebyshev_coeff.size(); ++i) {
            // h' = ci * residual + A@h
            copy(levels.at(lv).outh, levels.at(lv).residual);
            spmv(levels.at(lv).outh, 1, levels.at(lv).A, levels.at(lv).h, chebyshev_coeff.at(i), buff);

            // copy(levels.at(lv).h, levels.at(lv).outh);
            levels.at(lv).h.swap(levels.at(lv).outh);
        }

        axpy(x, 1, levels.at(lv).h); // x += h
    }

    void setup_jacobi(float const omega, size_t const n) {
        smoother_type = 2;
        jacobi_omega = omega;
        jacobi_niter = n;
    }


    void setup_jacobi_v2(CSR<float>&A, size_t const n) {
        // smoother_type = 2;
        jacobi_niter = n;

        // calc Dinv@A
        Vec<float> Dinv;
        Vec<float> data_new;
        Vec<float> diag;
        Dinv.resize(A.nrows);
        data_new.resize(A.data.size());
        diag.resize(A.nrows);
        calc_DinvA_kernel<<<(A.nrows + 255) / 256, 256>>>(data_new.data(), A.data.data(), A.indices.data(), A.indptr.data(), A.nrows, diag.data());

        CSR<float> DinvA;
        DinvA.assign(data_new.data(), A.numnonz, A.indices.data(), A.numnonz, A.indptr.data(), A.nrows+1, A.nrows, A.ncols, A.numnonz);

        float DinvA_rho = calc_max_eig(DinvA);
        jacobi_omega = 1.0 / (DinvA_rho);
        cout<<"DinvA_rho: "<<DinvA_rho<<endl;
        cout<<"jacobi_omega: "<<jacobi_omega<<endl; 
    }

    void jacobi(int lv, Vec<float> &x, Vec<float> const &b) {
        Vec<float> x_old;
        x_old.resize(x.size());
        copy(x_old, x);
        for (int i = 0; i < jacobi_niter; ++i) {
            weighted_jacobi_kernel<<<(levels.at(lv).A.nrows + 255) / 256, 256>>>(x.data(), x_old.data(), b.data(), levels.at(lv).A.data.data(), levels.at(lv).A.indices.data(), levels.at(lv).A.indptr.data(), levels.at(lv).A.nrows, jacobi_omega);
            x.swap(x_old);
        }
    }

    void jacobi_cpu(int lv, Vec<float> &x, Vec<float> const &b) {
        // serial jacobi
        std::vector<float> x_host(x.size());
        std::vector<float> b_host(b.size());
        x.tohost(x_host);
        b.tohost(b_host);
        std::vector<float> data_host;
        std::vector<int> indices_host, indptr_host;
        levels.at(lv).A.tohost(data_host, indices_host, indptr_host);
        // cout<<"omega: "<<jacobi_omega<<endl;
        jacobi_serial(
            indptr_host.data(), indptr_host.size(),
            indices_host.data(), indices_host.size(),
            data_host.data(), data_host.size(),
            x_host.data(), x_host.size(),
            b_host.data(), b_host.size(),
            x_host.data(), x_host.size(),
            0, levels.at(lv).A.nrows, 1, jacobi_omega);
        x.assign(x_host.data(), x_host.size());
        // auto r = calc_rnorm(b, x, levels.at(lv).A);
        // cout<<"lv"<<lv<<"   rnorm: "<<r<<endl;
    }


    void setup_gauss_seidel() {
        smoother_type = 3;
    }

    void gauss_seidel_cpu(int lv, Vec<float> &x, Vec<float> const &b) {
        // serial gauss seidel
        std::vector<float> x_host(x.size());
        std::vector<float> b_host(b.size());
        x.tohost(x_host);
        b.tohost(b_host);
        std::vector<float> data_host;
        std::vector<int> indices_host, indptr_host;
        levels.at(lv).A.tohost(data_host, indices_host, indptr_host);
        gauss_seidel_serial(
            indptr_host.data(), indptr_host.size(),
            indices_host.data(), indices_host.size(),
            data_host.data(), data_host.size(),
            x_host.data(), x_host.size(),
            b_host.data(), b_host.size(),
            0, levels.at(lv).A.nrows, 1);
        x.assign(x_host.data(), x_host.size());
        // auto r = calc_rnorm(b, x, levels.at(lv).A);
        // cout<<"lv"<<lv<<"   rnorm: "<<r<<endl;
    }

    void _smooth(int lv, Vec<float> &x, Vec<float> const &b) {
        if(smoother_type == 1)
        {
            chebyshev(lv, x, b);
        }
        else if (smoother_type == 2)
        {
            jacobi_cpu(lv, x, b);
        }
        else if (smoother_type == 3)
        {
            gauss_seidel_cpu(lv, x, b);
        }
    }


    void calc_residual(int lv, Vec<float> &x, Vec<float> const &b) {
        copy(levels.at(lv).residual, b);
        spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x
    }


    void vcycle_down() {
        for (int lv = 0; lv < nlvs-1; ++lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : z;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : r;

            _smooth(lv, x, b);

            copy(levels.at(lv).residual, b);
            spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x

            levels.at(lv).b.resize(levels.at(lv).R.nrows);
            spmv(levels.at(lv).b, 1, levels.at(lv).R, levels.at(lv).residual, 0, buff); // coarse_b = R@residual

            levels.at(lv).x.resize(levels.at(lv).b.size());
            zero(levels.at(lv).x);
        }
    }

    void vcycle_up() {
        for (int lv = nlvs-2; lv >= 0; --lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : z;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : r;
            spmv(x, 1, levels.at(lv).P, levels.at(lv).x, 1, buff); // x += P@coarse_x
            _smooth(lv, x, b);
        }
    }

    void vcycle() {
        vcycle_down();
        coarse_solve();
        vcycle_up();
    }


    void coarse_solve() {
        auto const &A = levels.at(nlvs - 1).A;
        auto &x = levels.at(nlvs - 2).x;
        auto &b = levels.at(nlvs - 2).b;
        spsolve(x, A, b);
    }

    void set_outer_x(float const *x, size_t n) {
        outer_x.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
        copy(x_new, outer_x);
    }

    void set_outer_b(float const *b, size_t n) {
        outer_b.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_b.data(), b, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    float init_cg_iter0(float *residuals) {
        float bnrm2 = vnorm(outer_b);
        // r = b - A@(x)
        copy(r, outer_b);
        spmv(outer_b, -1, levels.at(0).A, outer_x, 1, buff);
        float normr = vnorm(r);
        residuals[0] = normr;
        return bnrm2;
    }

    void do_cg_itern(float *residuals, size_t iteration) {
        float rho_cur = vdot(r, z);
        if (iteration > 0) {
            float beta = rho_cur / save_rho_prev;
            // p *= beta
            // p += z
            scal(save_p, beta);
            axpy(save_p, 1, z);
        } else {
            // p = move(z)
            save_p.swap(z);
        }
        // q = A@(p)
        save_q.resize(levels.at(0).A.nrows);
        spmv(save_q, 1, levels.at(0).A, save_p, 0, buff);
        save_rho_prev = rho_cur;
        float alpha = rho_cur / vdot(save_p, save_q);
        // x += alpha*p
        axpy(x_new, alpha, save_p);
        // r -= alpha*q
        axpy(r, -alpha, save_q);
        float normr = vnorm(r);
        residuals[iteration + 1] = normr;
    }

    void compute_RAP(size_t lv) {
            CSR<float> &A = levels.at(lv).A;
            CSR<float> &R = levels.at(lv).R;
            CSR<float> &P = levels.at(lv).P;
            CSR<float> AP;
            CSR<float> &RAP = levels.at(lv+1).A;
            R.resize(P.ncols, P.nrows, P.numnonz);
            transpose(P, R);            
            spgemm(A, P, AP) ;
            spgemm(R, AP, RAP);
    }

    void fetch_A(size_t lv, float *data, int *indices, int *indptr) {
        CSR<float> &A = levels.at(lv).A;
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    void set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol_, size_t maxiter_)
    {
        set_outer_x(x, nx);
        set_outer_b(b, nb);
        rtol = rtol_;
        maxiter = maxiter_;
        residuals.resize(maxiter+1);
    }

    float calc_max_eig(CSR<float>& A)
    {
        return  computeMaxEigenvaluePowerMethodOptimized(A, 100);
    }

    size_t get_data(float* x_out, float* r_out)
    {
        CHECK_CUDA(cudaMemcpy(x_out, x_new.data(), x_new.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(residuals.begin(), residuals.end(), r_out);
        return niter;
    }

    float sum(std::vector<float> &v)
    {
        return std::accumulate(v.begin(), v.end(), 0.0);
    }

    float avg(std::vector<float> &v)
    {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    void solve()
    {
        float bnrm2 = init_cg_iter0(residuals.data());
        float atol = bnrm2 * rtol;
        for (size_t iter=0; iter<maxiter; iter++)
        {   
            if (residuals[iter] < atol)
            {
                niter = iter;
                break;
            }
            copy(z, outer_x);
            vcycle();
            do_cg_itern(residuals.data(), iter); 
            niter = iter;
        }
    }
};

} // namespace


static VCycle *fastmg = nullptr;
static FastFillCloth *fastFillCloth = nullptr;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT void fastmg_new() {
    if (!fastmg)
        fastmg = new VCycle{};
}

extern "C" DLLEXPORT void fastmg_setup_nl(size_t numlvs) {
    fastmg->setup(numlvs);
}

extern "C" DLLEXPORT void fastmg_setup_jacobi(float const omega, size_t const niter_jacobi) {
    fastmg->setup_jacobi(omega, niter_jacobi);
}

extern "C" DLLEXPORT void fastmg_setup_gauss_seidel() {
    fastmg->setup_gauss_seidel();
}


extern "C" DLLEXPORT void fastmg_RAP(size_t lv) {
    fastmg->compute_RAP(lv);
}

extern "C" DLLEXPORT void fastmg_fetch_A(size_t lv, float* data, int* indices, int* indptr) {
    fastmg->fetch_A(lv, data, indices, indptr);
}


extern "C" DLLEXPORT void fastmg_solve() {
    fastmg->solve();
}

extern "C" DLLEXPORT void fastmg_set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol, size_t maxiter) {
    fastmg->set_data(x, nx, b, nb, rtol, maxiter);
}

extern "C" DLLEXPORT size_t fastmg_get_data(float *x, float *r) {
    size_t niter = fastmg->get_data(x, r);
    return niter;
}

extern "C" DLLEXPORT void fastmg_set_A0(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastmg->set_A0(data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}

// only update the data of A0
extern "C" DLLEXPORT void fastmg_update_A0(const float* data_in)
{
    fastmg->update_A0(data_in);
}

extern "C" DLLEXPORT void fastmg_set_P(int lv, float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastmg->set_P(lv, data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}


extern "C" DLLEXPORT void fastmg_setup_smoothers(int type) {
    fastmg->setup_smoothers_cuda(type);
}


extern "C" DLLEXPORT void fastmg_set_A0_from_fastFill() {
    fastmg->set_A0_from_fastFill(fastFillCloth);
}


// ------------------------------------------------------------------------------
extern "C" DLLEXPORT void fastFill_new() {
    if (!fastFillCloth)
        fastFillCloth = new FastFillCloth{};
}

extern "C" DLLEXPORT void fastFill_set_data(int* edges_in, int NE_in, float* inv_mass_in, int NV_in, float* pos_in, float alpha_in)
{
    fastFillCloth->set_data_v2(edges_in, NE_in, inv_mass_in, NV_in, pos_in, alpha_in);
}


extern "C" DLLEXPORT void fastFill_init_from_python_cache(
    int *adjacent_edge_in,
    int *num_adjacent_edge_in,
    int *adjacent_edge_abc_in,
    int num_nonz_in,
    float *spmat_data_in,
    int *spmat_indices_in,
    int *spmat_indptr_in,
    int *spmat_ii_in,
    int *spmat_jj_in,
    int NE_in,
    int NV_in)
{
    fastFillCloth->init_from_python_cache_v2(adjacent_edge_in,
                                     num_adjacent_edge_in,
                                     adjacent_edge_abc_in,
                                     num_nonz_in,
                                     spmat_data_in,
                                     spmat_indices_in,
                                     spmat_indptr_in,
                                     spmat_ii_in,
                                     spmat_jj_in,
                                     NE_in,
                                     NV_in);
}

extern "C" DLLEXPORT void fastFill_run(float* pos_in) {
    fastFillCloth->run(pos_in);
}

extern "C" DLLEXPORT void fastFill_fetch_A(float* data, int* indices, int* indptr) {
    fastFillCloth->fetch_A(data, indices, indptr);
}