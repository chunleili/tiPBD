#pragma once
#include "common.h"
#include "SpMatData.h"

struct LinearSolver
{
    float rtol = 1e-5;
    int maxiter = 100;
    std::vector<float> residuals;
    std::vector<float> solution;
    int niter;
    
    virtual Field1f& run(SpMatData* A, Field1f& b, bool should_setup=true)=0;
};


struct EigenSolver:LinearSolver
{
    virtual Field1f& run(SpMatData* A, Field1f& b, bool should_setup=true);
};

struct AmgclSolver:LinearSolver
{
    std::vector<SpMatData> Ps;
    virtual Field1f& run(SpMatData* A, Field1f& b, bool should_setup=true);
};




// struct AmgCuda:LinearSolver
// {
//     FastMG *fastmg;
//     AmgCuda(){fastmg = new FastMG();}

//     void run(SpMatData& A, Field1f& b, bool should_setup=true)
//     {
//         if (should_setup)
//         {
//             fastmg->setup();
//         }

//         fastmg->set_A0(A.data, A.nnz, A.indices, A.nnz, A.indptr, A.nrows + 1, A.nrows, A.ncols, A.nnz);
//         x0.clear()
//         x0.resize(b.size())
//         fastmg->set_data(x0.data(), x0.size(), b.data(), b.size(), rtol, maxiter);
//         fastmg->solve();

//         x.clear();
//         x.resize(b.size());
//         residuals.resize(maxiter);
//         niter = fastmg->get_data(x.data(), residuals.data())
//         niter+=1;
//         residuals.resize(niter);

//         printf("    inner iter: %d", niter);
//         printf("    residual: %.6e->%.6e",residuals[0], residuals[niter-1]);
//     }
// };
