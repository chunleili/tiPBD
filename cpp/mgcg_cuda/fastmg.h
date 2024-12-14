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
#include "mglevel.h"
#include "smoother.h"

using std::cout;
using std::endl;

#define USE_LESSMEM 1

namespace fastmg {


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

struct FastMG : CusparseWrappers {
    std::vector<MGLevel> levels; // create in create_levels
    std::unique_ptr<Smoother> smoother;  // create in create_levels

    size_t nlvs;
    size_t coarse_solver_type = 1; //0:direct solver by cusolver (cholesky), 1: one sweep smoother
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
    bool verbose = false;
    GpuTimer timer1,timer2,timer3;
    std::vector<float> elapsed1, elapsed2, elapsed3;


    void create_levels(size_t numlvs);
    void set_scale_RAP(float s, int lv);
    float calc_residual_norm(Vec<float> const &b, Vec<float> const &x, CSR<float> const &A);
    void set_P(size_t lv, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz);
    void set_A0(float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz);
    int get_nnz(int lv);
    int get_nrows(int lv);

    // only update the data of A0
    void update_A0(float const *datap) ;
    void set_A0_from_fastFill(FastFillBase *ff);
    void get_Aoff_and_Dinv(CSR<float> &A, CSR<float> &Dinv, CSR<float> &Aoff);

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
    size_t get_data(float* x_out, float* r_out);
    void presolve();

    void solve();
    void solve_only_jacobi();
    void solve_only_directsolver();
    void solve_only_smoother();
};



struct FastMG;
extern FastMG *fastmg;

} // namespace

