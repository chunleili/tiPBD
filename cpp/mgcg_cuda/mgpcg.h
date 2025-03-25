#pragma once

#include "cusparse_wrappers.h"
#include "Vec.h"
#include "CSR.h"
#include "timer.h"
#include "vcycle.h"
#include "config_manager.h"

namespace fastmg {

struct MGPCG : CusparseWrappers {
    std::vector<MGLevel>& levels; 
    std::shared_ptr<Smoother> smoother;  
    std::shared_ptr<VCycle> vcycle;  
    std::shared_ptr<ConfigManager> config;

    MGPCG(std::vector<MGLevel> &levels,
          std::shared_ptr<Smoother> smoother,
          std::shared_ptr<VCycle> vcycle,
          std::shared_ptr<ConfigManager> config);
    void solve(int maxiter, float rtol);
    void solve_only_smoother(int maxiter, float rtol);
    int niter;
    Vec<float> outer_x;
    Vec<float> x_new;
    Vec<float> outer_b;
    std::vector<float> residuals;
    float rtol;
    size_t maxiter;
    
private:
    float save_rho_prev;
    Vec<float> save_p;
    Vec<float> save_q;
    float init_cg_iter0(std::vector<float>& residuals);
    void do_cg_itern(std::vector<float>& residuals, size_t iteration);
    float calc_residual(CSR<float> const &A, Vec<float> &x, Vec<float> const &b, Vec<float> &r);






};

} // namespace

