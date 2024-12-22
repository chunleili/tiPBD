#include "linear_solver.h"
#include "SpMatData.h"
#include "common.h"

#include <vector>
#include <iostream>

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

Field1f& AmgclSolver::run(SpMatData* A_, Field1f& b, bool should_setup)
{
    // The profiler:
    amgcl::profiler<> prof("poisson3Db");

    int rows = A_->nrows();
    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, A_->indptr, A_->indices, A_->data);

    // Compose the solver type
    //   the solver backend:
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
    auto Ps = solve.precond().get_Ps();

    // Solve the system with the zero initial approximation:
    int iters;
    float error;
    std::vector<float> x(rows, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(A, b, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
    solution = std::move(x);
    return solution;
}


Field1f& EigenSolver::run(SpMatData* hostA, Field1f& b, bool should_setup)
{
    int nrows = hostA->nrows();
    int ncols = hostA->ncols();
    int nnz = hostA->nnz();
    // https://www.eigen.tuxfamily.org/dox/classEigen_1_1Map_3_01SparseMatrixType_01_4.html
    Eigen::Map<EigenSpMat> eigenA(
        nrows, ncols, nnz, hostA->indptr.data(), hostA->indices.data(), hostA->data.data());
    Eigen::SimplicialLLT<EigenSpMat,Eigen::Lower|Eigen::Upper> solver;
    solver.compute(eigenA);
    Map<VectorXf> b_(b.data(), b.size());
    Eigen::VectorXf x = solver.solve(b_);
    if(solver.info()!=Eigen::Success) {
        std::cerr << "Solver failed" << std::endl;
    }
    std::cout << "The solution is:\n" << x << std::endl;
    vector<float> x_(x.data(), x.data() + x.rows() );
    solution = std::move(x_);
    return solution;
}

