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

#include "amgcl_solver.h"
#include "common.h"
#include "SpMatData.h"
#include "linear_solver.h"


Field1f AmgclSolver::solve(SpMatData* A_, Field1f& b)
{
    // The profiler:
    amgcl::profiler<> prof("amgclSolver");

    int rows = A_->nrows();
    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, A_->indptr, A_->indices, A_->data);


    // if(should_setup)
    // {
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
        Solver setted_solver(A);
        prof.toc("setup");

        // Show the mini-report on the constructed solver:
        if(verbose)
            std::cout << setted_solver << std::endl;

        // output the prolongation operator:
        auto levels = setted_solver.precond().get_levels();
        size_t numlevels = levels.size();
        auto Ps = setted_solver.precond().get_Ps();
    // }



    // Solve the system with the zero initial approximation:
    int iters;
    float error;
    solution.clear();
    solution.resize(rows, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = setted_solver(A, b, solution);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "  Linear solver iters: " << iters 
              << "\terror: " << error << std::endl;
    if(verbose)
        std::cout << prof << std::endl;
    residuals.push_back(error);
    niter = iters;
    return (solution);
}