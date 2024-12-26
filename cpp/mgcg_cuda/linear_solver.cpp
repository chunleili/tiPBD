#include "linear_solver.h"
#include "SpMatData.h"
#include "common.h"
#include <unsupported/Eigen/SparseExtra>

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


#include "eigen/unsupported/Eigen/SparseExtra"
#include "mmio.h"

template<typename T = EigenSpMat>
 void saveMatrix(T& d, std::string filename = "mat")
 {
     Eigen::saveMarket(d, filename);
 }





Eigen::Matrix<float,-1,-1,RowMajor>
 spmat_to_dense(SpMatData* spA)
{
    int nrows = spA->nrows();
    int ncols = spA->ncols();
    int nnz = spA->nnz();
    Eigen::Matrix<float,-1,-1,RowMajor> dense(nrows, ncols);
    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<ncols; j++)
        {
            dense(i,j) = 0.0;
        }
    }
    for(int i=0; i<nrows; i++)
    {
        for(int j=spA->indptr[i]; j<spA->indptr[i+1]; j++)
        {
            dense(i, spA->indices[j]) = spA->data[j];
        }
    }
    return dense;
}



void AmgclSolver::run(SpMatData* A_, Field1f& b, bool should_setup)
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
    // std::cout << solve << std::endl;

    // output the prolongation operator:
    auto levels = solve.precond().get_levels();
    size_t numlevels = levels.size();
    auto Ps = solve.precond().get_Ps();

    // Solve the system with the zero initial approximation:
    int iters;
    float error;
    solution.clear();
    solution.resize(rows, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(A, b, solution);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl;
            //   << prof << std::endl;
}


void EigenSolver::run(SpMatData* hostA, Field1f& b, bool should_setup)
{
    int nrows = hostA->nrows();
    int ncols = hostA->ncols();
    int nnz = hostA->nnz();

    // transfer the A to Eigen
    // // https://www.eigen.tuxfamily.org/dox/classEigen_1_1Map_3_01SparseMatrixType_01_4.html
    // Eigen::Map<EigenSpMat> eigenA(
    //     nrows, ncols, nnz, hostA->indptr.data(), hostA->indices.data(), hostA->data.data());
    // Workaround for Eigen bug, deep copy the data
    EigenSpMat eigenA = EigenSpMat(nrows, ncols);
    eigenA.reserve(nnz);
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(nrows);
    for(int i = 0; i < nrows; i++)
    {
        for(int j = hostA->indptr[i]; j < hostA->indptr[i+1]; j++)
        {
            tripletList.push_back(T(i, hostA->indices[j], hostA->data[j]));
        }
    }
    eigenA.setFromTriplets(tripletList.begin(), tripletList.end());

    Map<VectorXf> b_(b.data(), b.size());
    // saveMatrix(eigenA, "eigenA.mtx");
    // savetxt(b_, "b.mtx");

    Eigen::SimplicialLLT<EigenSpMat> solver;
    solver.compute(eigenA);
    Eigen::VectorXf x = solver.solve(b_);
    if(solver.info()!=Eigen::Success) {
        std::cerr << "Solver failed" << std::endl;
    }
    std::cout << "The solution is:\n" << x << std::endl;
    std::copy(x.data(), x.data()+x.size(), solution.begin());
}

