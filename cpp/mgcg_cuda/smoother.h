#pragma once

#include "cusparse_wrappers.h"
#include "mglevel.h"
#include "timer.h"

namespace fastmg
{


struct Smoother:CusparseWrappers{
    Smoother() = default;
    Smoother(std::vector<MGLevel> &levels) : levels(levels) {} //pass the reference of levels in fastmg to the smoother

    std::vector<float> chebyshev_coeff;
    size_t smoother_type = 1; //1:chebyshev, 2:w-jacobi, 3:gauss_seidel(level0)+w-jacobi(other levels)
    size_t smoother_niter=2; // TODO: we will replace smoother_niter later
    float m_max_eig=0.0;
    bool use_radical_omega=false;
    Buffer buff;

    std::vector<float> m_elapsed;
    GpuTimer m_timer;


    void setup_smoothers(int type);
    void set_smoother_niter(size_t const n);
    void smooth(int lv, Vec<float> &x, Vec<float> const &b);
    void set_colors(const int* c, int n, int color_num_in, int lv);
    
    void jacobi(int lv, Vec<float> &x, Vec<float> const &b);
    void jacobi_v2(int lv, Vec<float> &x, Vec<float> const &b);//  cusparse version
    // void jacobi_cpu(int lv, Vec<float> &x, Vec<float> const &b);
    void gauss_seidel_cpu(int lv, Vec<float> &x, Vec<float> const &b);
    void multi_color_gauss_seidel(int lv, Vec<float> &x, Vec<float> const &b);

    std::vector<MGLevel> &levels; // reference to the levels in fastmg
    
    void setup_chebyshev_cuda(CSR<float> &A);
    void chebyshev_polynomial_coefficients(float a, float b);
    void chebyshev(int lv, Vec<float> &x, Vec<float> const &b);
    void setup_weighted_jacobi();
    float calc_min_eig(CSR<float> &A, float mu0=0.1);
    float calc_weighted_jacobi_omega(CSR<float>&A, bool use_radical_omega=false);
    float calc_max_eig(CSR<float>& A);
    float get_max_eig();
};
    
} // namespace fastmg