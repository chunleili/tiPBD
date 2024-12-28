#include "eigen_solver.h"

template<typename T = EigenSpMat>
 void saveMatrix(T& d, std::string filename = "mat")
 {
     Eigen::saveMarket(d, filename);
 }


Field1f EigenSolver::solve(SpMatData* hostA, Field1f& b)
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
    return solution;
}


