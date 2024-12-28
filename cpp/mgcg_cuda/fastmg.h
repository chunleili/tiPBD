#pragma once

#include "cusparse_wrappers.h"
#include "Vec.h"
#include "CSR.h"
#include "timer.h"
#include "mgpcg.h"
#include "SpMatData.h"

namespace fastmg {
template <typename T>
struct Vec;
template <typename T>
struct CSR;
struct MGLevel;
struct VCycle;
struct Smoother;
struct FastFillBase;



struct FastMG :CusparseWrappers{
    std::vector<MGLevel> levels; // create in create_levels
    std::shared_ptr<Smoother> smoother;  // create in create_levels
    std::shared_ptr<VCycle> vcycle;  // create in create_levels
    std::shared_ptr<MGPCG> mgpcg;  // create in create_levels


    bool verbose = false;
    GpuTimer timer1,timer2,timer3;
    std::vector<float> elapsed1, elapsed2, elapsed3;

    void setup(SpMatData* P0); //setup P0 from outside
    void create_levels(size_t numlvs);

    void set_scale_RAP(float s, int lv);
    void set_P(size_t lv, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz);

    void set_A0(float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz);
    void update_A0(float const *datap) ;// only update the data of A0
    void set_A0_from_fastFill(FastFillBase *ff);
    void fetch_A_data(float *data);
    void fetch_A(size_t lv, float *data, int *indices, int *indptr);
    int get_nnz(int lv);
    int get_nrows(int lv);

    void compute_RAP(size_t lv);
    void set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol_, size_t maxiter_);
    size_t get_data(float* x_out, float* r_out);
    void presolve();

    void solve();
    void solve_only_jacobi();
    void solve_only_directsolver();
    void solve_only_smoother();

    FastMG(){};
    static FastMG* get_instance() {
        static FastMG instance;
        return &instance;
    }

private:
    void get_Aoff_and_Dinv(CSR<float> &A, CSR<float> &Dinv, CSR<float> &Aoff);
    void set_outer_x(float const *x, size_t n);
    void set_outer_b(float const *b, size_t n);
};

FastMG* get_fastmg();

} // namespace

