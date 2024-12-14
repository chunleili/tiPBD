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
#include "cusparse_wrappers.h"

using std::cout;
using std::endl;

#define USE_LESSMEM 1
#define VERBOSE 0

namespace fastmg {




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

struct VCycle : CusparseWrappers {
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

