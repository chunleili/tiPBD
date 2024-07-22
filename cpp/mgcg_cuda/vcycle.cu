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
#include <filesystem>
#include <array>
#include <unordered_set>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/SparseExtra"

using namespace std;
using EigenSpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;


#if __GNUC__ && __linux__
#include <sys/ptrace.h>

[[noreturn]] static void cuerr() {
    if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) != 0)
        __builtin_trap();
    exit(EXIT_FAILURE);
}
#elif _WIN32 && _MSC_VER
#include <windows.h>

[[noreturn]] static void cuerr() {
    int debugger_present = 0;
    HANDLE process = GetCurrentProcess();
    CheckRemoteDebuggerPresent(process, &debugger_present);
    if (debugger_present) {
        __debugbreak();
    }
    exit(EXIT_FAILURE);
}
#else
[[noreturn]] static void cuerr() {
    exit(EXIT_FAILURE);
}
#endif

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("%s:%d: %s (%d): %s\n", __FILE__, __LINE__,                     \
               cudaGetErrorString(status), status, #func);                     \
        cuerr();                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("%s:%d: %s (%d): %s\n", __FILE__, __LINE__,                     \
               cusparseGetErrorString(status), status, #func);                 \
        cuerr();                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("%s:%d: %s (%d): %s\n", __FILE__, __LINE__,                     \
               cublasGetStatusString(status), status, #func);                  \
        cuerr();                                                               \
    }                                                                          \
}


// https://github.com/NVIDIA/CUDALibrarySamples/blob/ed19a07b6dd0900b7547b274a6ed9d7c22a6d431/cuSOLVER/utils/cusolver_utils.h#L55
#define CHECK_CUSOLVER(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)


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

    void assign(T const *data, size_t size) {
        resize(size);
        CHECK_CUDA(cudaMemcpy(m_data, data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    void store(T *data) const {
        CHECK_CUDA(cudaMemcpy(data, m_data, sizeof(T) * size(), cudaMemcpyDeviceToHost));
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

template <class T>
struct CSR {
    Vec<T> data;
    Vec<int> indices;
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
};

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
        ConstSpMat matA(matA_);
        ConstSpMat matB(matB_);
        matC_.resize(matA_.nrows, matB_.ncols, 0);
        SpMat matC(matC_);
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
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize1, NULL) )
        // CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
        dBuffer1.reserve(bufferSize1);

        // inspect the matrices A and B to understand the memory requirement for
        // the next step
        CHECK_CUSPARSE(
            cusparseSpGEMM_workEstimation(cusparse, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize1, dBuffer1.data()) )

        // ask bufferSize2 bytes for external memory
        CHECK_CUSPARSE(
            cusparseSpGEMM_compute(cusparse, opA, opB,
                                &alpha, matA, matB, &beta, matC,
                                computeType, CUSPARSE_SPGEMM_DEFAULT,
                                spgemmDesc, &bufferSize2, NULL) )
        // CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )
        dBuffer2.reserve(bufferSize2);

        // compute the intermediate product of A * B
        CHECK_CUSPARSE( cusparseSpGEMM_compute(cusparse, opA, opB,
                                            &alpha, matA, matB, &beta, matC,
                                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc, &bufferSize2, dBuffer2.data()) )
        // --------------------------------------------------------------------------
        // get matrix C non-zero entries C_nnz1
        CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &matC_.nrows, &matC_.ncols, &matC_.numnonz) )
        // allocate matrix C
        matC_.resize(matC_.nrows, matC_.ncols, matC_.numnonz);
        // update matC with the new pointers
        CHECK_CUSPARSE(cusparseCsrSetPointers(matC, matC_.indptr.data(), matC_.indices.data(), matC_.data.data()) )

        // copy the final products to the matrix C
        CHECK_CUSPARSE(
            cusparseSpGEMM_copy(cusparse, opA, opB,
                                &alpha, matA, matB, &beta, matC,
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
    void spsolve(Vec<float> &x, CSR<float> const &A, Vec<float> &b) {
        // https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csrlsvchol
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


    void transpose(CSR<float> const & A, CSR<float>& AT)
    {
        // https://docs.nvidia.com/cuda/cusparse/index.html?highlight=cusparseCsr2cscEx2#cusparsecsr2cscex2

        // cusparseHandle_t     handle = NULL;
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

    // https://stackoverflow.com/a/57382195/19253199
    void CuSparseToEigenSparse(CSR<float> const &A, EigenSpMat &mat) 
    {
        //EigenSpMat is RowMajor, i.e. CSR
        const int *indptr = A.indptr.data();
        const int *indices = A.indices.data();
        const float *data = A.data.data();
        const int nnz = A.numnonz;
        const int nrows = A.nrows;
        const int ncols = A.ncols;
        std::vector<int> inner(nnz);       // inner index is the column indices: indices
        std::vector<int> outer(nrows + 1); // outer index is the rowStart: indptr
        std::vector<float> value(nnz);    // value

        cudaMemcpy(inner.data(), indices, sizeof(int) * nnz,         cudaMemcpyDeviceToHost);
        cudaMemcpy(outer.data(), indptr,  sizeof(int) * (nrows + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(value.data(), data,    sizeof(float) * nnz,       cudaMemcpyDeviceToHost);

        Eigen::Map<EigenSpMat> mat_map(
            nrows, ncols, nnz, outer.data(), inner.data(), value.data());

        mat = mat_map.eval();
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

struct VCycle : Kernels {
    std::vector<MGLevel> levels;
    size_t nlvs;
    std::vector<float> coefficients;
    Vec<float> init_x;
    Vec<float> init_b;
    Vec<float> outer_x;
    Vec<float> alter_x;
    Vec<float> outer_b;
    float save_rho_prev;
    Vec<float> save_p;
    Vec<float> save_q;
    Buffer buff;
    float rtol;
    size_t maxiter;
    std::vector<float> residuals;
    size_t niter; //final number of iterations to break the loop

    void setup(size_t numlvs) {
        if (levels.size() < numlvs) {
            levels.resize(numlvs);
        }
        nlvs = numlvs;
        coefficients.clear();
    }

    void set_lv_csrmat(size_t lv, size_t which, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        CSR<float> *mat = nullptr;
        if (which == 1) mat = &levels.at(lv).A;
        if (which == 2) mat = &levels.at(lv).R;
        if (which == 3) mat = &levels.at(lv).P;
        if (mat) {
            mat->assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
        }
    }

    void set_coeff(float const *coeff, size_t ncoeffs) {
        coefficients.assign(coeff, coeff + ncoeffs);
    }

    void _smooth(int lv, Vec<float> &x, Vec<float> const &b) {
        copy(levels.at(lv).residual, b);
        spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x
        scal2(levels.at(lv).h, coefficients.at(0), levels.at(lv).residual); // h = c0 * residual


        for (int i = 1; i < coefficients.size(); ++i) {
            // h' = ci * residual + A@h
            copy(levels.at(lv).outh, levels.at(lv).residual);
            spmv(levels.at(lv).outh, 1, levels.at(lv).A, levels.at(lv).h, coefficients.at(i), buff);

            // copy(levels.at(lv).h, levels.at(lv).outh);
            levels.at(lv).h.swap(levels.at(lv).outh);
        }

        axpy(x, 1, levels.at(lv).h); // x += h
    }

    void set_init_x(float const *x, size_t n) {
        init_x.resize(n);
        CHECK_CUDA(cudaMemcpy(init_x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void set_init_b(float const *b, size_t n) {
        init_b.resize(n);
        CHECK_CUDA(cudaMemcpy(init_b.data(), b, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void vcycle_down() {
        for (int lv = 0; lv < nlvs-1; ++lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : init_x;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : init_b;
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
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : init_x;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : init_b;
            spmv(x, 1, levels.at(lv).P, levels.at(lv).x, 1, buff); // x += P@coarse_x
            _smooth(lv, x, b);
        }
    }

    void vcycle() {
        vcycle_down();
        coarse_solve();
        vcycle_up();
    }

    size_t get_coarsist_size() {
        auto const &this_b = levels.at(nlvs - 2).b;
        return this_b.size();
    }

    void get_coarsist_b(float *b) {
        auto const &this_b = levels.at(nlvs - 2).b;
        CHECK_CUDA(cudaMemcpy(b, this_b.data(), this_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void get_finest_x(float *x) {
        CHECK_CUDA(cudaMemcpy(x, init_x.data(), init_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void set_coarsist_x(float const *x) {
        auto const &this_b = levels.at(nlvs - 2).b;
        auto &this_x = levels.at(nlvs - 2).x;
        this_x.resize(this_b.size());
        CHECK_CUDA(cudaMemcpy(this_x.data(), x, this_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void coarse_solve() {
        auto const &A = levels.at(nlvs - 1).A;
        auto &x = levels.at(nlvs - 2).x;
        auto &b = levels.at(nlvs - 2).b;
        spsolve(x, A, b);
    }

    void copy_outer2init_x() {
        copy(init_x, outer_x);
    }

    void set_outer_x(float const *x, size_t n) {
        outer_x.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
        copy(alter_x, outer_x);
    }

    void set_outer_b(float const *b, size_t n) {
        outer_b.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_b.data(), b, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    float init_cg_iter0(float *residuals) {
        float bnrm2 = vnorm(outer_b);
        // r = b - A@(x)
        copy(init_b, outer_b);
        spmv(outer_b, -1, levels.at(0).A, outer_x, 1, buff);
        float normr = vnorm(init_b);
        residuals[0] = normr;
        return bnrm2;
    }

    void do_cg_itern(float *residuals, size_t iteration) {
        float rho_cur = vdot(init_b, init_x);
        if (iteration > 0) {
            float beta = rho_cur / save_rho_prev;
            // p *= beta
            // p += z
            scal(save_p, beta);
            axpy(save_p, 1, init_x);
        } else {
            // p = move(z)
            save_p.swap(init_x);
        }
        // q = A@(p)
        save_q.resize(levels.at(0).A.nrows);
        spmv(save_q, 1, levels.at(0).A, save_p, 0, buff);
        save_rho_prev = rho_cur;
        float alpha = rho_cur / vdot(save_p, save_q);
        // x += alpha*p
        axpy(alter_x, alpha, save_p);
        // r -= alpha*q
        axpy(init_b, -alpha, save_q);
        float normr = vnorm(init_b);
        residuals[iteration + 1] = normr;
    }

    void fetch_cg_final_x(float *x) {
        CHECK_CUDA(cudaMemcpy(x, alter_x.data(), alter_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void fetch_cg_final_r(float *r) {
        // CHECK_CUDA(cudaMemcpy(r, residuals.data(), residuals.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(residuals.begin(), residuals.end(), r);
    }

    void compute_RAP(size_t lv) {
            CSR<float> &A = levels.at(lv).A;
            CSR<float> &R = levels.at(lv).R;
            CSR<float> &P = levels.at(lv).P;
            CSR<float> AP;
            CSR<float> &RAP = levels.at(lv+1).A;
            spgemm(A, P, AP) ;
            spgemm(R, AP, RAP);
    }

    void fetch_A(size_t lv, float *data, int *indices, int *indptr) {
        CSR<float> &A = levels.at(lv).A;
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    void set_mgcg_data(const float* x, size_t nx, const float* b, size_t nb, float rtol_, size_t maxiter_)
    {
        set_outer_x(x, nx);
        set_outer_b(b, nb);
        rtol = rtol_;
        maxiter = maxiter_;
        residuals.resize(maxiter+1);
    }

    size_t get_mgcg_data(float* x_, float* r_)
    {
        fetch_cg_final_x(x_);
        fetch_cg_final_r(r_);
        return niter;
    }

    void mgcg_solve()
    {
        float bnrm2 = init_cg_iter0(residuals.data());
        float atol = bnrm2 * rtol;
        for (size_t iteration=0; iteration<maxiter; iteration++)
        {   
            if (residuals[iteration] < atol)
            {
                niter = iteration; //number of iter to break
                break;
            }
            copy_outer2init_x();  //reset x to x0
            vcycle();
            do_cg_itern(residuals.data(), iteration); //first r is r[0], then r[iter+1]
        }
    }


};

struct AssembleMatrix : Kernels {
    CSR<float> A;
    CSR<float> G;
    CSR<float> M;
    CSR<float> ALPHA;
    float alpha;
    int NE;

    void fetch_A(float *data, int *indices, int *indptr) {
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    void set_G(float const *datap, int const *indicesp, int const *indptrp, int rows, int cols, int nnz) {
        G.assign(datap, nnz, indicesp, nnz, indptrp, rows + 1, rows, cols, nnz);
    }

    void set_M(float const *datap, int const *indicesp, int const *indptrp, int rows, int cols, int nnz) {
        M.assign(datap, nnz, indicesp, nnz, indptrp, rows + 1, rows, cols, nnz);
    }

    void set_ALPHA(float const *datap, int const *indicesp, int const *indptrp, int rows, int cols, int nnz) {
        ALPHA.assign(datap, nnz, indicesp, nnz, indptrp, rows + 1, rows, cols, nnz);
    }

    void compute_GMG() {
        CSR<float> GM;
        spgemm(G, M, GM);
        CSR<float> GT;
        GT.resize(G.ncols, G.nrows, G.numnonz);
        transpose(G, GT);
        spgemm(GM, GT, A);
    }

};

} // namespace

static VCycle *fastmg = nullptr;
static AssembleMatrix *fastA = nullptr;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT void fastmg_setup(size_t numlvs) {
    if (!fastmg)
        fastmg = new VCycle{};
    fastmg->setup(numlvs);
}

extern "C" DLLEXPORT void fastmg_set_coeff(float const *coeff, size_t ncoeffs) {
    fastmg->set_coeff(coeff, ncoeffs);
}

extern "C" DLLEXPORT void fastmg_set_lv_csrmat(size_t lv, size_t which, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
    fastmg->set_lv_csrmat(lv, which, datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
}

extern "C" DLLEXPORT void fastmg_RAP(size_t lv) {
    fastmg->compute_RAP(lv);
}

extern "C" DLLEXPORT void fastmg_fetch_A(size_t lv, float* data, int* indices, int* indptr) {
    fastmg->fetch_A(lv, data, indices, indptr);
}

extern "C" DLLEXPORT void fastmg_vcycle() {
    fastmg->vcycle();
}

extern "C" DLLEXPORT void fastmg_mgcg_solve() {
    fastmg->mgcg_solve();
}

extern "C" DLLEXPORT void fastmg_set_mgcg_data(const float* x, size_t nx, const float* b, size_t nb, float rtol, size_t maxiter) {
    fastmg->set_mgcg_data(x, nx, b, nb, rtol, maxiter);
}

extern "C" DLLEXPORT size_t fastmg_get_mgcg_data(float *x, float *r) {
    size_t niter = fastmg->get_mgcg_data(x, r);
    return niter;
}

// ------------------------------------------------------------------------------
extern "C" DLLEXPORT void fastA_setup() {
    if (!fastA)
        fastA = new AssembleMatrix{};
}

extern "C" DLLEXPORT void fastA_set_G(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastA->set_G(data, indices, indptr, rows, cols, nnz);
}

extern "C" DLLEXPORT void fastA_set_M(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastA->set_M(data, indices, indptr, rows, cols, nnz);
}

extern "C" DLLEXPORT void fastA_set_ALPHA(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastA->set_ALPHA(data, indices, indptr, rows, cols, nnz);
}

extern "C" DLLEXPORT void fastA_compute_GMG() {
    fastA->compute_GMG();
}

extern "C" DLLEXPORT void fastA_fetch_A(float* data, int* indices, int* indptr) {
    fastA->fetch_A(data, indices, indptr);
}