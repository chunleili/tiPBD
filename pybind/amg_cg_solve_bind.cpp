#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tuple>
#include <vector>

using namespace std;
using Eigen::Map;
using Eigen::Vector3f;
using Eigen::VectorXf;
using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;


class MultiLevel
{
public:
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> P;
    Eigen::SparseMatrix<double> R;
};



template<class I, class T, class F>
void gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                  const T  b[], const int  b_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}


// An easy-to-use wrapper for gauss_seidel
void easy_gauss_seidel(const SpMat &A_, const VectorXd &b_, VectorXd &x_)
{
    int max_GS_iter = 1;
    std::fill(x_.begin(), x_.end(), 0.0);
    for (int GS_iter = 0; GS_iter < max_GS_iter; GS_iter++)
    {
        gauss_seidel<int, double, double>(A_.outerIndexPtr(), A_.outerSize(),
                                        A_.innerIndexPtr(), A_.innerSize(), A_.valuePtr(), A_.nonZeros(),
                                        x_.data(), x_.size(), b_.data(), b_.size(), 0, b_.size(), 1);
    }
}

Eigen::VectorXd coarse_solver(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b)
{
    // use eigen direct solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Eigen::VectorXd x = solver.solve(b);
    return x;
}


void V_cycle(std::vector<MultiLevel>& levels, int lvl, Eigen::VectorXd& x, Eigen::VectorXd& b)
{
    Eigen::SparseMatrix<double> A = levels[lvl].A;
    // gauss_seidel(A, x, b, 1, 'symmetric');
    easy_gauss_seidel(A, b, x);
    Eigen::VectorXd residual = b - A * x;
    Eigen::VectorXd coarse_b = levels[lvl].R * residual;
    Eigen::VectorXd coarse_x = Eigen::VectorXd::Zero(coarse_b.size());
    if (lvl == levels.size() - 2)
    {
        coarse_x = coarse_solver(levels[lvl + 1].A, coarse_b);
    }
    else
    {
        V_cycle(levels, lvl + 1, coarse_x, coarse_b);
    }
    x += levels[lvl].P * coarse_x;
    // gauss_seidel(A, x, b, 1, 'symmetric');
    easy_gauss_seidel(A, b, x);
}



Eigen::VectorXd psolve(Eigen::VectorXd& b, std::vector<MultiLevel>& levels)
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    V_cycle(levels, 0, x, b);
    return x;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> amg_cg_solve_bind(std::vector<MultiLevel>& levels, Eigen::VectorXd& b, Eigen::VectorXd& x0, double tol, int maxiter)
{
    Eigen::VectorXd x = x0;
    Eigen::SparseMatrix<double> A = levels[0].A;
    Eigen::VectorXd residuals(maxiter + 1);
    Eigen::VectorXd r = b - A * x;
    double bnrm2 = b.norm();
    double atol = tol * bnrm2;
    double rho_prev;
    Eigen::VectorXd p;
    double normr = r.norm();
    residuals(0) = normr;
    for (int iteration = 0; iteration < maxiter; iteration++)
    {
        if (normr < atol)
        {
            break;
        }
        Eigen::VectorXd z = psolve(r, levels);
        double rho_cur = r.dot(z);
        if (iteration > 0)
        {
            double beta = rho_cur / rho_prev;
            p *= beta;
            p += z;
        }
        else
        {
            p = z;
        }
        Eigen::VectorXd q = A * p;
        double alpha = rho_cur / p.dot(q);
        x += alpha * p;
        r -= alpha * q;
        rho_prev = rho_cur;
        normr = r.norm();
        residuals(iteration + 1) = normr;
    }
    return std::make_tuple(x, residuals);
}



// #include <pybind11/pybind11.h>
// #include <iostream>
// namespace py = pybind11;


// PYBIND11_MODULE(AmgPybind, m) {
//     m.def("amg_cg_solve_bind", &amg_cg_solve_bind, "amg_cg_solve_bind");
// }



#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;

//绑定一个函数
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(amg_cg_solve_bind, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    // m.def("add", &add, "A function that adds two numbers");
    m.def("amg_cg_solve_bind", &amg_cg_solve_bind, "amg_cg_solve_bind");
}