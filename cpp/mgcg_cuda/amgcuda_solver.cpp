#include "fastmg.h"
#include "amgcuda_solver.h"
#include "mglevel.h"

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include <memory>

namespace fastmg
{

Field1f AmgCudaSolver::solve(SpMatData* A, Field1f& b, bool should_setup)
{
    FastMG* fastmg = FastMG::get_instance();

    if (should_setup)
    {
        this->setup(A);
        fastmg->setup(m_Ps);
    }

    fastmg->set_A0(A->data.data(), A->nnz(), A->indices.data(), A->nnz(), A->indptr.data(), A->nrows()+1, A->nrows(), A->ncols(), A->nnz());
    std::vector<float> x0(A->nrows(), 0.0);
    fastmg->set_data(x0.data(), x0.size(), b.data(), b.size(), rtol, maxiter);
    fastmg->solve();

    solution.resize(A->nrows());
    residuals.resize(maxiter);
    niter = fastmg->get_data(solution.data(), residuals.data());
    niter+=1;
    residuals.resize(niter);

    printf("    inner iter: %d", niter);
    printf("    residual: %.6e->%.6e",residuals[0], residuals[niter-1]);
    return solution;
}


std::vector<SpMatData*> AmgCudaSolver::setup(SpMatData* A_)
{
    // The profiler:
    amgcl::profiler<> prof("amgclSolver");

    int rows = A_->nrows();
    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, A_->indptr, A_->indices, A_->data);

    typedef amgcl::backend::builtin<float> SBackend;
    typedef amgcl::backend::builtin<float> PBackend;
    typedef amgcl::make_solver<
        amgcl::amg<
            PBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::damped_jacobi
            >,
        amgcl::solver::cg<SBackend>
        > Solver;

    // Set the parameters for the solver:
    Solver::params prm;
    prm.precond.coarsening.relax = 0.0; //Unsmoothed aggregation
    prm.precond.coarsening.aggr.eps_strong = 0.25; //Strength threshold
    
    // Initialize the solver with the system matrix:
    prof.tic("setup");
    Solver solve(A);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // output the prolongation operator:
    auto levels = solve.precond().get_levels();
    size_t numlevels = levels.size();
    auto Ps_ = solve.precond().get_Ps();

    return m_Ps;
}


} // namespace fastmg
