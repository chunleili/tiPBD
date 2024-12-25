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


// #include "fastmg.h"

// namespace fastmg
// {
// struct AmgCuda:LinearSolver
// {
//     AmgCuda(){}

//     Field1f& run(SpMatData* A, Field1f& b, bool should_setup=true)
//     {
//         FastMG* fastmg = FastMG::get_instance();
//         if (should_setup)
//         {
//             fastmg->setup();
//         }
// // float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz
//         fastmg->set_A0(A->data.data(), A->nnz(), A->indices.data(), A->nnz(), A->indptr.data(), A->nrows()+1, A->nrows(), A->ncols(), A->nnz());
//         std::vector<float> x0(A->nrows(), 0.0);
//         fastmg->set_data(x0.data(), x0.size(), b.data(), b.size(), rtol, maxiter);
//         fastmg->solve();

//         solution.resize(A->nrows());
//         residuals.resize(maxiter);
//         niter = fastmg->get_data(solution.data(), residuals.data());
//         niter+=1;
//         residuals.resize(niter);

//         printf("    inner iter: %d", niter);
//         printf("    residual: %.6e->%.6e",residuals[0], residuals[niter-1]);
//     }
// };
// }