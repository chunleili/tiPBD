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

// Terminal output color (just for cosmetic purpose)
#define RST  "\x1B[37m"  // Reset color to white
#define KGRN  "\033[0;32m"   // Define green color
#define RD "\x1B[31m"  // Define red color
#define FGRN(x) KGRN x RST  // Define compiler function for green color
#define FRD(x) RD x RST  // Define compiler function for red color

using namespace std;

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



// Generate random number in the range [0, 1)
struct genRandomNumber {
    __host__ __device__
    float operator()(const int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }
};



/// @brief Usage: Timer t("timer_name");
///               t.start();
///               //do something
///               t.end();
/// You need to include <chrono> and <string> for this to work
class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
    std::chrono::duration<double, std::milli> elapsed_ms;
    std::chrono::duration<double> elapsed_s;
    double elapsed=0.0;
    std::string name = "";

    Timer(std::string name = "") : name(name){};
    inline void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    inline void end(std::string msg = "", std::string unit = "ms", bool verbose=true, std::string endsep = "\n")
    {
        m_end = std::chrono::steady_clock::now();
        if (unit == "s")
        {
            elapsed_s = m_end - m_start;
            if(verbose)
                printf("%s(%s): %.0f(s)", msg.c_str(), name.c_str(), elapsed_s.count());
            else
                printf("%.0f(s)", elapsed_s.count());
        }
        else //else if(unit == "ms")
        {
            elapsed_ms = m_end - m_start;
            if(verbose)
                printf("%s(%s): %.0f(ms)", msg.c_str(), name.c_str(), elapsed_ms.count());
            else
                printf("%.0f(ms)", elapsed_ms.count());
        }
        printf("%s", endsep.c_str());
    }
    inline void reset()
    {
        m_start = std::chrono::steady_clock::now();
        m_end = std::chrono::steady_clock::now();
        elapsed = 0.0;
    };
    inline void accumulate()
    {
        m_end = std::chrono::steady_clock::now();
        elapsed += std::chrono::duration<double, std::milli>(m_end - m_start).count();
    };
    inline void report()
    {
        std::cout << name << ": " << elapsed << " ms" << std::endl;
    };
    
};


// https://stackoverflow.com/a/41154786/19253199
// https://github.com/aramadia/udacity-cs344/blob/master/Unit2%20Code%20Snippets/gputimer.h
// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
/// @brief Usage: 
///     GpuTimer  timer;
///     timer.start(); 
///     do something
///     timer.stop(); 
///     float elapsedTime = timer.elapsed(); 
///     printf("Elapsed time : %.2f ms\n" ,elapsedTime);
struct GpuTimer
{
      cudaEvent_t m_start;
      cudaEvent_t m_stop;

      GpuTimer()
      {
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_stop);
      }

      ~GpuTimer()
      {
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
      }

      void start()
      {
            cudaEventRecord(m_start, 0);
      }

      void stop()
      {
            cudaEventRecord(m_stop, 0);
      }

      float elapsed()
      {
            float elapsed_ms;
            cudaEventSynchronize(m_stop);
            cudaEventElapsedTime(&elapsed_ms, m_start, m_stop);
            return elapsed_ms;
      }
};



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



/* -------------------------------------------------------------------------- */
/*                                   kernels                                  */
/* -------------------------------------------------------------------------- */



__device__ float3 inline d_normalize_diff(std::array<float,3> &v1,  std::array<float,3> &v2)
{
    std::array<float,3> diff = {v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]};
    float norm = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
    return {diff[0]/norm, diff[1]/norm, diff[2]/norm};
}

__device__ float inline d_dot(std::array<float,3> a, std::array<float,3> b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}



// def fill_A_CSR_kernel(data:ti.types.ndarray(dtype=ti.f32), 
//                               indptr:ti.types.ndarray(dtype=ti.i32), 
//                               ii:ti.types.ndarray(dtype=ti.i32), 
//                               jj:ti.types.ndarray(dtype=ti.i32),
//                               adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),
//                               num_nonz:ti.i32,
//                               alpha:ti.f32):
//     for cnt in range(num_nonz):
//         i = ii[cnt] # row index
//         j = jj[cnt] # col index
//         k = cnt - indptr[i] #k-th non-zero element of i-th row. 
//         # Because the diag is the final element of each row, 
//         # it is also the k-th adjacent edge of i-th edge.
//         if i == j: # diag
//             data[cnt] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha
//             continue
//         a = adjacent_edge_abc[i, k * 3]
//         b = adjacent_edge_abc[i, k * 3 + 1]
//         c = adjacent_edge_abc[i, k * 3 + 2]
//         g_ab = (pos[a] - pos[b]).normalized()
//         g_ac = (pos[a] - pos[c]).normalized()
//         offdiag = inv_mass[a] * g_ab.dot(g_ac)
//         data[cnt] = offdiag
// __global__ void fill_A_CSR_kernel(thrust::device_vector<float> &data, 
//                                   thrust::device_vector<int> indptr, 
//                                   thrust::device_vector<int> ii, 
//                                   thrust::device_vector<int> jj,
//                                   thrust::device_vector<thrust::device_vector<int>> adjacent_edge_abc,
//                                   int num_nonz,
//                                   float alpha,
//                                   thrust::device_vector<float3> pos,
//                                   thrust::device_vector<float> inv_mass) {
//     size_t cnt = blockIdx.x * blockDim.x + threadIdx.x;
//     if (cnt < num_nonz) {
//         int i = ii[cnt]; // row index
//         int j = jj[cnt]; // col index
//         int k = cnt - indptr[i]; //k-th non-zero element of i-th row. 
//         // Because the diag is the final element of each row, 
//         // it is also the k-th adjacent edge of i-th edge.
//         if (i == j) { // diag
//             data[cnt] = inv_mass[i] + inv_mass[i] + alpha;
//             return;
//         }
//         int a = adjacent_edge_abc[i][k * 3];
//         int b = adjacent_edge_abc[i][k * 3 + 1];
//         int c = adjacent_edge_abc[i][k * 3 + 2];
//         float3 g_ab = d_normalize_diff(pos[a], pos[b]);
//         float3 g_ac = d_normalize_diff(pos[a], pos[c]);
//         float offdiag = inv_mass[a] * d_dot(g_ab, g_ac);
//         data[cnt] = offdiag;
//     }
// }




// weighted Jacobi for csr matrix
// https://en.wikipedia.org/wiki/Jacobi_method#Weighted_Jacobi_method
// https://stackoverflow.com/questions/78057439/jacobi-algorithm-using-cuda
// https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/amg_core/relaxation.h#L232
// i: row index, j: col index, n: data/indices index
// rsum: sum of off-diagonal elements
__global__ void weighted_jacobi_kernel(float *x, float *x_old, const float *b, float *data, int *indices, int *indptr, int nrows, float omega) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        float rsum = 0.0;
        float diag = 0.0;
        for (size_t n = indptr[i]; n < indptr[i + 1]; ++n) {
            size_t j = indices[n];
            if (j != i) {
                rsum += data[n] * x_old[j];
            }
            else {
                diag = data[n];
            }
        }
        // FIXME: should use x_new to avoid race condition
        if (diag != 0.0)
        {
            x[i] =  omega / diag * (b[i] - rsum)  + (1.0 - omega) * x_old[i];
        }
    }
}

__global__ void copy_field(float *dst, const float *src, int size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] = src[i];
    }
}


void jacobi_serial(const int Ap[], const int Ap_size,
            const int Aj[], const int Aj_size,
            const float Ax[], const int Ax_size,
                  float  x[], const int  x_size,
            const float  b[], const int  b_size,
                  float temp[], const int temp_size,
            const int row_start,
            const int row_stop,
            const int row_step,
            const float omega)
{
    float one = 1.0;

    for(int i = row_start; i != row_stop; i += row_step) {
        temp[i] = x[i];
    }

    for(int i = row_start; i != row_stop; i += row_step) {
        int start = Ap[i];
        int end   = Ap[i+1];
        float rsum = 0;
        float diag = 0;

        for(int jj = start; jj < end; jj++){
            int j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*temp[j];
        }

        if (diag != (float) 0.0){
            x[i] = (one - omega) * temp[i] + omega * ((b[i] - rsum)/diag);
        }
    }
}



// https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/amg_core/relaxation.h#L45
void gauss_seidel_serial(const int Ap[], const int Ap_size,
                  const int Aj[], const int Aj_size,
                  const float Ax[], const int Ax_size,
                        float  x[], const int  x_size,
                  const float  b[], const int  b_size,
                  const int row_start,
                  const int row_stop,
                  const int row_step)
{
    for(int i = row_start; i != row_stop; i += row_step) {
        int start = Ap[i];
        int end   = Ap[i+1];
        float rsum = 0;
        float diag = 0;

        for(int jj = start; jj < end; jj++){
            int j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (float) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
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
  float tol = 1e-6;  // tolerance for convergence
  int itr = 0;
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

    if (std::abs(max_eigenvalue - max_eigenvalue_prev) < tol) {
      std::cout << FGRN("[NOTE]: ") << "Converged at iterations: " << itr << std::endl;
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

  std::cout << FRD("[NOTE]: ") << "Maximum number of iterations reached." << std::endl;  // no convergence
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


using thrust::device_vector;

__global__ void fill_A_CSR_kernel(float *a, int size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] = 100.0;
    }
}


struct FastFill : Kernels {
    CSR<float> A;
    float alpha;
    int NE;
    int NV;
    std::vector<std::array<int,2>> edges;
    std::vector<float> inv_mass;
    std::vector<std::array<float,3>> pos;
    std::vector<std::vector<int>> adjacent_edges;
    std::vector<int> num_adjacent_edge;
    std::vector<std::vector<int>> adjacent_edge_abc;
    std::vector<int> ii, jj;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> data;
    int num_nonz;
    int nrows, ncols;
    Vec<float> d_inv_mass;

    void fetch_A(float *data_in, int *indices_in, int *indptr_in) {
        std::copy(data.begin(), data.end(), data_in);
        std::copy(indices.begin(), indices.end(), indices_in);
        std::copy(indptr.begin(), indptr.end(), indptr_in);
    }

    void set_data(int* edges_in, int NE_in, float* inv_mass_in, int NV_in, float* pos_in, float alpha_in)
    {
        NE = NE_in;
        NV = NV_in;
        nrows = NE;
        ncols = NE;

        edges.resize(NE);
        for(int i=0; i<NE; i++)
        {
            edges[i][0] = edges_in[i*2];
            edges[i][1] = edges_in[i*2+1];
        }

        inv_mass.resize(NV_in);
        for(int i=0; i<NV_in; i++)
        {
            inv_mass[i] = inv_mass_in[i];
        }

        pos.resize(NV);
        for(int i=0; i<NV; i++)
        {
            pos[i][0] = pos_in[i*3];
            pos[i][1] = pos_in[i*3+1];
            pos[i][2] = pos_in[i*3+2];
        }

        alpha = alpha_in;
    }

    void update_pos(float* pos_in)
    {
        for(int i=0; i<NV; i++)
        {
            pos[i][0] = pos_in[i*3];
            pos[i][1] = pos_in[i*3+1];
            pos[i][2] = pos_in[i*3+2];
        }
    }

    void host_to_device()
    {
        d_inv_mass.assign(inv_mass.data(), inv_mass.size());
        cout<<"copy data to device"<<endl;
    }

    void device_to_host()
    {
        d_inv_mass.tohost(inv_mass);
    }


    // init_direct_fill_A
    int init()
    {
        init_adj_edge(edges);
        init_adjacent_edge_abc();
        calc_num_nonz();
        init_A_CSR_pattern();
        csr_index_to_coo_index();

        // transfer data to device
        host_to_device();

        return num_nonz;
    }

    void run(float* pos_in)
    {
        Timer t;
        t.start();
        update_pos(pos_in);
        t.end("update_pos");
        t.start();
        fill_A_CSR_gpu();
        t.end("fill_A_CSR");
    }


    std::array<float,3> inline normalize(std::array<float,3> v)
    {
        float norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        return {v[0]/norm, v[1]/norm, v[2]/norm};
    }

    std::array<float,3> inline normalize_diff(std::array<float,3> &v1,  std::array<float,3> &v2)
    {
        std::array<float,3> diff = {v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]};
        float norm = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        return {diff[0]/norm, diff[1]/norm, diff[2]/norm};
    }

    float inline dot(std::array<float,3> a, std::array<float,3> b)
    {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    void launch_check()
    {
        cudaError_t varCudaError1 = cudaGetLastError();
        if (varCudaError1 != cudaSuccess)
        {
            std::cout << "Failed to launch kernel (error code: " << cudaGetErrorString(varCudaError1) << ")!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void fill_A_CSR_gpu()
    {
        TODO: finish fill A CSR
        fill_A_CSR_kernel<<<NV/128+1, 128>>>(d_inv_mass.data(), NV);
        cudaDeviceSynchronize();
        launch_check();

        cout<<"111"<<endl;
        device_to_host();
        cout<<"inv_mass[0]: "<<inv_mass[0]<<endl;
        exit(0);
    }


    void fill_A_CSR()
    {
        for(int cnt=0; cnt<num_nonz; cnt++)
        {
            int i = ii[cnt]; // row index
            int j = jj[cnt]; // col index
            int k = cnt - indptr[i]; //k-th non-zero element of i-th row. 
            // Because the diag is the final element of each row, 
            // it is also the k-th adjacent edge of i-th edge.
            if (i == j) // diag
            {
                data[cnt] = inv_mass[edges[i][0]] + inv_mass[edges[i][1]] + alpha;
                continue;
            }
            int a = adjacent_edge_abc[i][k*3];
            int b = adjacent_edge_abc[i][k*3+1];
            int c = adjacent_edge_abc[i][k*3+2];
            auto g_ab = normalize_diff(pos[a], pos[b]);
            auto g_ac = normalize_diff(pos[a], pos[c]);
            auto offdiag = inv_mass[a] * dot(g_ab, g_ac);
            data[cnt] = offdiag;
        }
    }


    void init_A_CSR_pattern()
    {
        indptr.resize(NE+1);
        indices.resize(num_nonz);
        data.resize(num_nonz);

        indptr[0] = 0;
        for(int i=0; i<NE; i++)
        {
            int num_adj_i = num_adjacent_edge[i];
            indptr[i+1] = indptr[i] + num_adj_i + 1;
            for(int j=0; j<num_adj_i; j++)
            {
                indices[indptr[i]+j] = adjacent_edges[i][j];
            }
            indices[indptr[i+1]-1] = i;
        }
    }


    void csr_index_to_coo_index()
    {
        ii.resize(num_nonz);
        jj.resize(num_nonz);
        for(int i=0; i<NE; i++)
        {
            for(int j=indptr[i]; j<indptr[i+1]; j++)
            {
                ii[j] = i;
                jj[j] = indices[j];
            }
        }
    }


    void init_adj_edge(std::vector<std::array<int,2>> &edges)
    {
        std::unordered_map<int, std::set<int>> vertex_to_edges;
        for(int edge_index=0; edge_index<edges.size(); edge_index++)
        {
            int v1 = edges[edge_index][0];
            int v2 = edges[edge_index][1];
            if (vertex_to_edges.find(v1) == vertex_to_edges.end())
                vertex_to_edges[v1] = std::set<int>();
            if (vertex_to_edges.find(v2) == vertex_to_edges.end())
                vertex_to_edges[v2] = std::set<int>();
            vertex_to_edges[v1].insert(edge_index);
            vertex_to_edges[v2].insert(edge_index);
        }

        adjacent_edges.resize(edges.size());
        for(int edge_index=0; edge_index<edges.size(); edge_index++)
        {
            int v1 = edges[edge_index][0];
            int v2 = edges[edge_index][1];
            std::set<int> adj; //adjacent edges of one edge
            std::set_union(vertex_to_edges[v1].begin(), vertex_to_edges[v1].end(), vertex_to_edges[v2].begin(), vertex_to_edges[v2].end(), std::inserter(adj, adj.begin()));
            adj.erase(edge_index);
            adjacent_edges[edge_index] = std::vector<int>(adj.begin(), adj.end());
        }

        //calc num_adjacent_edge
        for(auto adj:adjacent_edges)
        {
            num_adjacent_edge.push_back(adj.size());
        }

        NE = edges.size();

        adjacent_edge_abc.resize(NE);
        for(int i=0; i<NE; i++)
        {
            adjacent_edge_abc[i].resize(num_adjacent_edge[i]*3);
        }
    }

    void calc_num_nonz()
    {
        num_nonz = 0;
        for(auto num_adj:num_adjacent_edge)
        {
            num_nonz += num_adj;
        }
        num_nonz += num_adjacent_edge.size();

        A.numnonz = num_nonz;
    }


    void init_adjacent_edge_abc()
    {
        for(int i=0; i<edges.size(); i++)
        {
            auto ii0 = edges[i][0];
            auto ii1 = edges[i][1];

            auto num_adj = num_adjacent_edge[i];
            for(int j=0; j<num_adj; j++)
            {
                auto ia = adjacent_edges[i][j];
                if (ia == i)
                    continue;
                auto jj0 = edges[ia][0];
                auto jj1 = edges[ia][1];
                auto a = -1;
                auto b = -1;
                auto c = -1;
                if (ii0 == jj0)
                {
                    a = ii0;
                    b = ii1;
                    c = jj1;
                }
                else if (ii0 == jj1)
                {
                    a = ii0;
                    b = ii1;
                    c = jj0;
                }
                else if (ii1 == jj0)
                {
                    a = ii1;
                    b = ii0;
                    c = jj1;
                }
                else if (ii1 == jj1)
                {
                    a = ii1;
                    b = ii0;
                    c = jj0;
                }
                adjacent_edge_abc[i][j*3] = a;
                adjacent_edge_abc[i][j*3+1] = b;
                adjacent_edge_abc[i][j*3+2] = c;
            }
        }
    }

}; //FastFill struct


struct VCycle : Kernels {
    std::vector<MGLevel> levels;
    size_t nlvs;
    std::vector<float> chebyshev_coeff;
    size_t smoother_type = 1; //1:chebyshev, 2:jacobi, 3:gauss_seidel
    float jacobi_omega;
    size_t jacobi_niter;
    Vec<float> init_x;
    Vec<float> init_b;
    Vec<float> outer_x;
    Vec<float> final_x;
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
        if(smoother_type == 1)
        {
            setup_chebyshev_cuda(levels[0].A);
        }
        else if (smoother_type == 2)
        {
            //TODO:setup jacobi
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

    // DEPRECATED
    // void set_lv_csrmat(size_t lv, size_t which, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
    //     CSR<float> *mat = nullptr;
    //     if (which == 1) mat = &levels.at(lv).A;
    //     if (which == 2) mat = &levels.at(lv).R;
    //     if (which == 3) mat = &levels.at(lv).P;
    //     if (mat) {
    //         mat->assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    //     }
    // }

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


    void set_A0_from_fastFill(FastFill *ff) {
        levels.at(0).A.assign(ff->data.data(), ff->data.size(), ff->indices.data(), ff->indices.size(), ff->indptr.data(), ff->indptr.size(), ff->nrows, ff->ncols, ff->num_nonz);
    }

    // DEPRECATED
    // void setup_chebyshev(float const *coeff, size_t ncoeffs) {
    //     smoother_type = 1;
    //     chebyshev_coeff.assign(coeff, coeff + ncoeffs);
    // }

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
            chebyshev(lv, x, b);
        else if (smoother_type == 2)
        {
            jacobi(lv, x, b);
        }
        else if (smoother_type == 3)
        {
            gauss_seidel_cpu(lv, x, b);
        }
    }

    GpuTimer ttt1, ttt2, ttt3, ttt;
    std::vector<float> ttt1_elapsed, ttt2_elapsed, ttt3_elapsed;
    std::vector<std::vector<float>> ttt_elapsed;


    void calc_residual(int lv, Vec<float> &x, Vec<float> const &b) {
        copy(levels.at(lv).residual, b);
        spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x
    }


    void vcycle_down() {
        ttt_elapsed.resize(nlvs-1);
        for (int lv = 0; lv < nlvs-1; ++lv) {
            ttt.start();

            ttt1.start();
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : init_x;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : init_b;
            ttt1.stop();
            ttt1_elapsed.push_back(ttt1.elapsed());

            ttt2.start();
            _smooth(lv, x, b);
            ttt2.stop();
            ttt2_elapsed.push_back(ttt2.elapsed());

            ttt3.start();
            copy(levels.at(lv).residual, b);
            spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x

            levels.at(lv).b.resize(levels.at(lv).R.nrows);
            spmv(levels.at(lv).b, 1, levels.at(lv).R, levels.at(lv).residual, 0, buff); // coarse_b = R@residual

            levels.at(lv).x.resize(levels.at(lv).b.size());
            zero(levels.at(lv).x);
            ttt3.stop();
            ttt3_elapsed.push_back(ttt3.elapsed());

            ttt.stop();
            ttt_elapsed[lv].push_back(ttt.elapsed());
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

    GpuTimer tt1, tt2, tt3;
    std::vector<float> tt1_elapsed, tt2_elapsed, tt3_elapsed;

    void vcycle() {
        
        tt1.start();
        vcycle_down();
        tt1.stop();
        tt1_elapsed.push_back(tt1.elapsed());

        tt2.start();
        coarse_solve();
        tt2.stop();
        tt2_elapsed.push_back(tt2.elapsed());


        tt3.start();
        vcycle_up();
        tt3.stop();
        tt3_elapsed.push_back(tt3.elapsed());
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
        copy(final_x, outer_x);
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
        axpy(final_x, alpha, save_p);
        // r -= alpha*q
        axpy(init_b, -alpha, save_q);
        float normr = vnorm(init_b);
        residuals[iteration + 1] = normr;
    }

    void fetch_cg_final_x(float *x) {
        CHECK_CUDA(cudaMemcpy(x, final_x.data(), final_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
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

    float get_max_eig()
    {
        Timer t("eigenvalue");
        t.start();
        max_eig = computeMaxEigenvaluePowerMethodOptimized(levels.at(0).A, 100);
        t.end();
        cout<<"max eigenvalue: "<<max_eig<<endl;
        return  max_eig;
    }

    size_t get_data(float* x_, float* r_)
    {
        fetch_cg_final_x(x_);
        fetch_cg_final_r(r_);
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
        GpuTimer t1, t2, t3, t4, t5;
        std::vector<float> t2_elapsed, t3_elapsed, t4_elapsed, t5_elapsed;

        t1.start();
        float bnrm2 = init_cg_iter0(residuals.data());
        float atol = bnrm2 * rtol;
        for (size_t iter=0; iter<maxiter; iter++)
        {   
            t2.start();
            t3.start();

            if (residuals[iter] < atol)
            {
                niter = iter;
                break;
            }
            copy_outer2init_x();  //reset x to x0
            t3.stop();
            t3_elapsed.push_back(t3.elapsed());

            t4.start();
            vcycle();
            t4.stop();
            t4_elapsed.push_back(t4.elapsed());

            t5.start();
            do_cg_itern(residuals.data(), iter); //first r is r[0], then r[iter+1]
            t5.stop();
            t5_elapsed.push_back(t5.elapsed());

            niter = iter;
            t2.stop();
            t2_elapsed.push_back(t2.elapsed());
        }

        bool report_time = false;
        if(report_time)
        {
            float avg_t2 = avg(t2_elapsed);
            float avg_t3 = avg(t3_elapsed);
            float avg_t4 = avg(t4_elapsed);
            float avg_t5 = avg(t5_elapsed);
            float avg_tt1 = avg(tt1_elapsed);
            float avg_tt2 = avg(tt2_elapsed);
            float avg_tt3 = avg(tt3_elapsed);
            float avg_ttt1 = avg(ttt1_elapsed);
            float avg_ttt2 = avg(ttt2_elapsed);
            float avg_ttt3 = avg(ttt3_elapsed);
            

            cout<<"     avg time one iteration: "<<avg_t2<<" ms"<<endl;
            cout<<"     avg time before vcycle: "<<avg_t3<<" ms"<<endl;
            cout<<"     avg time vcycle: "<<avg_t4<<" ms"<<endl;
            cout<<"     avg time after vcycle: "<<avg_t5<<" ms"<<endl;

            cout<<"     avg time vcycle_down: "<<avg_tt1<<" ms"<<endl;
            cout<<"     avg time coarse_solve: "<<avg_tt2<<" ms"<<endl;
            cout<<"     avg time vcycle_up: "<<avg_tt3<<" ms"<<endl;

            cout<<"     avg time vcycle_down before smooth: "<<avg_ttt1<<" ms"<<endl;
            cout<<"     avg time vcycle_down smooth: "<<avg_ttt2<<" ms"<<endl;
            cout<<"     avg time vcycle_down after smooth: "<<avg_ttt3<<" ms"<<endl;

            // print ttt elaspse
            for(int lv=0; lv<nlvs-1; lv++)
            {
                cout<<"     level "<<lv;
                cout<<" avg ttt time: "<< avg(ttt_elapsed[lv])<<" ms"<<endl;
            }

            t1.stop();
            cout<<"     time of solve: "<<t1.elapsed()<<" ms"<<endl;
        }
    }


};

// struct AssembleMatrix : Kernels {
//     CSR<float> A;
//     CSR<float> G;
//     CSR<float> M;
//     CSR<float> ALPHA;
//     float alpha;
//     int NE;

//     void fetch_A(float *data, int *indices, int *indptr) {
//         CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
//         CHECK_CUDA(cudaMemcpy(indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
//         CHECK_CUDA(cudaMemcpy(indptr, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
//     }

//     void set_G(float const *datap, int const *indicesp, int const *indptrp, int rows, int cols, int nnz) {
//         G.assign(datap, nnz, indicesp, nnz, indptrp, rows + 1, rows, cols, nnz);
//     }

//     void set_M(float const *datap, int const *indicesp, int const *indptrp, int rows, int cols, int nnz) {
//         M.assign(datap, nnz, indicesp, nnz, indptrp, rows + 1, rows, cols, nnz);
//     }

//     void set_ALPHA(float const *datap, int const *indicesp, int const *indptrp, int rows, int cols, int nnz) {
//         ALPHA.assign(datap, nnz, indicesp, nnz, indptrp, rows + 1, rows, cols, nnz);
//     }

//     void compute_GMG() {
//         CSR<float> GM;
//         spgemm(G, M, GM);
//         CSR<float> GT;
//         GT.resize(G.ncols, G.nrows, G.numnonz);
//         transpose(G, GT);
//         spgemm(GM, GT, A);
//     }

// };




} // namespace

static VCycle *fastmg = nullptr;
// static AssembleMatrix *fastA = nullptr;
static FastFill *fastFill = nullptr;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT void fastmg_new(size_t numlvs) {
    if (!fastmg)
        fastmg = new VCycle{};
}

extern "C" DLLEXPORT void fastmg_setup_nl(size_t numlvs) {
    fastmg->setup(numlvs);
}

// extern "C" DLLEXPORT void fastmg_setup_chebyshev(float const *coeff, size_t ncoeffs) {
//     fastmg->setup_chebyshev(coeff, ncoeffs);
// }

extern "C" DLLEXPORT void fastmg_setup_jacobi(float const omega, size_t const niter_jacobi) {
    fastmg->setup_jacobi(omega, niter_jacobi);
}

extern "C" DLLEXPORT void fastmg_setup_gauss_seidel() {
    fastmg->setup_gauss_seidel();
}

// extern "C" DLLEXPORT void fastmg_set_lv_csrmat(size_t lv, size_t which, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
//     fastmg->set_lv_csrmat(lv, which, datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
// }

extern "C" DLLEXPORT void fastmg_RAP(size_t lv) {
    fastmg->compute_RAP(lv);
}

extern "C" DLLEXPORT void fastmg_fetch_A(size_t lv, float* data, int* indices, int* indptr) {
    fastmg->fetch_A(lv, data, indices, indptr);
}

// extern "C" DLLEXPORT void fastmg_vcycle() {
//     fastmg->vcycle();
// }

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
                // data, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz
    fastmg->set_A0(data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}

// only update the data of A0
extern "C" DLLEXPORT void fastmg_update_A0(const float* data_in)
{
    fastmg->update_A0(data_in);
}

extern "C" DLLEXPORT void fastmg_set_P(int lv, float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
                //lv, data, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz
    fastmg->set_P(lv, data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}

extern "C" DLLEXPORT float fastmg_get_max_eig() {
    return fastmg->get_max_eig();
}

// extern "C" DLLEXPORT void fastmg_cheby_poly(float a, float b) {
//     fastmg->chebyshev_polynomial_coefficients(a, b);
// }

extern "C" DLLEXPORT void fastmg_setup_smoothers(int type) {
    fastmg->setup_smoothers_cuda(type);
}


extern "C" DLLEXPORT void fastmg_set_A0_from_fastFill() {
    fastmg->set_A0_from_fastFill(fastFill);
}

// // ------------------------------------------------------------------------------
// extern "C" DLLEXPORT void fastA_new() {
//     if (!fastA)
//         fastA = new AssembleMatrix{};
// }

// extern "C" DLLEXPORT void fastA_set_G(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
// {
//     fastA->set_G(data, indices, indptr, rows, cols, nnz);
// }

// extern "C" DLLEXPORT void fastA_set_M(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
// {
//     fastA->set_M(data, indices, indptr, rows, cols, nnz);
// }

// extern "C" DLLEXPORT void fastA_set_ALPHA(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
// {
//     fastA->set_ALPHA(data, indices, indptr, rows, cols, nnz);
// }

// extern "C" DLLEXPORT void fastA_compute_GMG() {
//     fastA->compute_GMG();
// }

// extern "C" DLLEXPORT void fastA_fetch_A(float* data, int* indices, int* indptr) {
//     fastA->fetch_A(data, indices, indptr);
// }

// ------------------------------------------------------------------------------
extern "C" DLLEXPORT void fastFill_new() {
    if (!fastFill)
        fastFill = new FastFill{};
}

extern "C" DLLEXPORT void fastFill_set_data(int* edges_in, int NE_in, float* inv_mass_in, int NV_in, float* pos_in, float alpha_in)
{
    fastFill->set_data(edges_in, NE_in, inv_mass_in, NV_in, pos_in, alpha_in);
}

// init_direct_fill_A
extern "C" DLLEXPORT int fastFill_init() {
    int nnz = fastFill->init();
    return nnz;
}


extern "C" DLLEXPORT void fastFill_run(float* pos_in) {
    fastFill->run(pos_in);
}

extern "C" DLLEXPORT void fastFill_fetch_A(float* data, int* indices, int* indptr) {
    fastFill->fetch_A(data, indices, indptr);
}