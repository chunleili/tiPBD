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

#include "kernels.cuh"
#include "cuda_utils.cuh"

using std::cout;
using std::endl;

#define USE_LESSMEM 1
#define VERBOSE 0

namespace fastmg {

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

    void assign_v2(T const *datap,  int const *indicesp,  int const *indptrp, size_t rows, size_t cols, size_t nnz) {
        int ndat = nnz;
        int nind = nnz;
        int nptr = rows + 1;
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



struct MGLevel {
    CSR<float> A;
    CSR<float> R;
    CSR<float> P;
    Vec<float> residual;
    Vec<float> b;
    Vec<float> x;
    Vec<float> h;
    Vec<float> outh;
    CSR<float> Dinv;
    CSR<float> Aoff;
    float scale_RAP=0.0;
    float jacobi_omega=2.0/3.0;
    std::array<float,3> chebyshev_coeff;
    Vec<int> colors; // color index of each node
    int color_num; // number of colors, max(colors)+1
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




template <typename T=float>
std::vector<T> debug_cuda_vec(Vec<T> &v, std::string name) {
    std::vector<T> v_host(v.size());
    v.tohost(v_host);
    cout<<name<<"("<<v.size()<<") :";
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




struct FastFillBase;

struct VCycle : Kernels {
    std::vector<MGLevel> levels;
    size_t nlvs;
    std::vector<float> chebyshev_coeff;
    size_t smoother_type = 1; //1:chebyshev, 2:w-jacobi, 3:gauss_seidel(level0)+w-jacobi(other levels)
    size_t coarse_solver_type = 1; //0:direct solver by cusolver (cholesky), 1: one sweep smoother
    size_t smoother_niter=2; // TODO: we will replace smoother_niter later
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
    bool use_radical_omega=true;
    bool verbose = false;
    GpuTimer timer1,timer2,timer3;
    std::vector<float> elapsed1, elapsed2, elapsed3;

    void set_scale_RAP(float s, int lv);
    void setup_smoothers(int type);
    void setup_chebyshev_cuda(CSR<float> &A);
    void chebyshev_polynomial_coefficients(float a, float b);
    float calc_residual_norm(Vec<float> const &b, Vec<float> const &x, CSR<float> const &A);
    void setup(size_t numlvs);
    void set_P(size_t lv, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz);
    void set_A0(float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz);
    int get_nnz(int lv);
    int get_nrows(int lv);

    // only update the data of A0
    void update_A0(float const *datap) ;
    void set_A0_from_fastFill(FastFillBase *ff);
    void chebyshev(int lv, Vec<float> &x, Vec<float> const &b);
    void set_smoother_niter(size_t const n);
    void setup_weighted_jacobi();
    float calc_min_eig(CSR<float> &A, float mu0=0.1);
    float calc_weighted_jacobi_omega(CSR<float>&A, bool use_radical_omega=false);

    void get_Aoff_and_Dinv(CSR<float> &A, CSR<float> &Dinv, CSR<float> &Aoff);
    void jacobi(int lv, Vec<float> &x, Vec<float> const &b);
    // use cusparse instead of hand-written kernel
    void jacobi_v2(int lv, Vec<float> &x, Vec<float> const &b);
    void gauss_seidel_cpu(int lv, Vec<float> &x, Vec<float> const &b);
    void set_colors(const int* c, int n, int color_num_in, int lv);
    void multi_color_gauss_seidel(int lv, Vec<float> &x, Vec<float> const &b);
    void _smooth(int lv, Vec<float> &x, Vec<float> const &b);
    float calc_residual(int lv, CSR<float> const &A, Vec<float> &x, Vec<float> const &b);
    void vcycle_down();
    void vcycle_up();
    void vcycle();
    void coarse_solve();
    void set_outer_x(float const *x, size_t n);
    void set_outer_b(float const *b, size_t n);
    float init_cg_iter0(float *residuals);
    void do_cg_itern(float *residuals, size_t iteration);
    void compute_RAP(size_t lv);
    void fetch_A_data(float *data);
    void fetch_A(size_t lv, float *data, int *indices, int *indptr);
    void set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol_, size_t maxiter_);
    float calc_max_eig(CSR<float>& A);
    size_t get_data(float* x_out, float* r_out);
    void presolve();

    void solve();
    void solve_only_jacobi();
    void solve_only_directsolver();
    void solve_only_smoother();
};






struct VCycle;
extern VCycle *fastmg;

} // namespace

