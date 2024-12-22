#include "mgpcg.h"
#include "mglevel.h"
#include "smoother.h"

namespace fastmg
{
    

void  MGPCG::solve(int maxiter, float rtol)
{
    residuals.resize(maxiter + 1);
    float bnrm2 = init_cg_iter0(residuals);
    float atol = bnrm2 * rtol;
    for (size_t iter=0; iter<maxiter; iter++)
    {   
        if (residuals[iter] < atol)
        {
            niter = iter;
            break;
        }
        copy(vcycle->z, outer_x);
        vcycle -> run();
        do_cg_itern(residuals, iter); 
        niter = iter;
    }
}

float  MGPCG::calc_residual(CSR<float> const &A, Vec<float> &x, Vec<float> const &b, Vec<float> &r) {
    copy(r, b);
    spmv(r, -1, A, x, 1, buff); // residual = b - A@x
    return vnorm(r);
}

void  MGPCG::solve_only_smoother(int maxiter, float rtol)
{
    residuals.resize(maxiter + 1);
    float bnrm2 = init_cg_iter0(residuals);
    float atol = bnrm2 * rtol;
    for (size_t iter=0; iter<maxiter; iter++)
    {   
        smoother->smooth(0, outer_x, outer_b);
        auto rnorm = calc_residual(levels.at(0).A, outer_x, outer_b, vcycle->r);
        residuals[iter] = rnorm;
        if (residuals[iter] < atol)
        {
            niter = iter;
            break;
        }
        niter = iter;
    }
    copy(x_new, outer_x);

}



void  MGPCG::do_cg_itern(std::vector<float> &residuals, size_t iteration) {
    float rho_cur = vdot(vcycle->r, vcycle->z);
    if (iteration > 0) {
        float beta = rho_cur / save_rho_prev;
        // p *= beta
        // p += z
        scal(save_p, beta);
        axpy(save_p, 1, vcycle->z);
    } else {
        // p = move(z)
        save_p.swap(vcycle->z);
    }
    // q = A@(p)
    save_q.resize(levels.at(0).A.nrows);
    spmv(save_q, 1, levels.at(0).A, save_p, 0, buff);
    save_rho_prev = rho_cur;
    float alpha = rho_cur / vdot(save_p, save_q);
    // x += alpha*p
    axpy(x_new, alpha, save_p);
    // r -= alpha*q
    axpy(vcycle->r, -alpha, save_q);
    float normr = vnorm(vcycle->r);
    residuals[iteration + 1] = normr;
}


float  MGPCG::init_cg_iter0(std::vector<float>& residuals) {
    float bnrm2 = vnorm(outer_b);
    // r = b - A@(x)
    copy(vcycle->r, outer_b);
    spmv(outer_b, -1, levels.at(0).A, outer_x, 1, buff);
    float normr = vnorm(vcycle->r);
    residuals[0] = normr;
    return bnrm2;
}

} // namespace fastmg
