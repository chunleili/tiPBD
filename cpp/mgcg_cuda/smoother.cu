#include "smoother.h"
#include "kernels.cuh"
#include "timer.h"

#include <iostream>
#include <vector>
#include <cassert>

using std::cout;
using std::endl;


namespace fastmg
{
    

void Smoother::setup_smoothers(int type)
{
    if (verbose)
        cout << "\nSetting up smoothers..." << endl;
    smoother_type = type;
    if (smoother_type == 1)
    {
        setup_chebyshev_cuda(levels[0].A);
    }
    else if (smoother_type == 2)
    {
        setup_weighted_jacobi();
    }
    else if (smoother_type == 3)
    {
        // TODO: multi-color GS for all levels
        setup_weighted_jacobi();
    }
}

void Smoother::setup_chebyshev_cuda(CSR<float> &A)
{
    float lower_bound = 1.0 / 30.0;
    float upper_bound = 1.1;
    float rho = computeMaxEigenvaluePowerMethodOptimized(A, 100);
    float a = rho * lower_bound;
    float b = rho * upper_bound;
    chebyshev_polynomial_coefficients(a, b);

    max_eig = rho;
    if (verbose)
    {
        cout << "max eigenvalue: " << max_eig << endl;
    }
}

void Smoother::chebyshev_polynomial_coefficients(float a, float b)
{
    int degree = 3;
    const float PI = 3.14159265358979323846;

    if (a >= b || a <= 0)
        assert(false && "Invalid input for Chebyshev polynomial coefficients");

    // Chebyshev roots for the interval [-1,1]
    std::vector<float> std_roots(degree);
    for (int i = 0; i < degree; i++)
    {
        std_roots[i] = std::cos(PI * (i + 0.5) / (float)degree);
    }

    // Chebyshev roots for the interval [a,b]
    std::vector<float> scaled_roots(degree);
    for (int i = 0; i < degree; i++)
    {
        scaled_roots[i] = 0.5 * (b - a) * (1 + std_roots[i]) + a;
    }

    // Compute monic polynomial coefficients of polynomial with scaled roots
    std::vector<float> scaled_poly(4);
    // np.poly for 3 roots. This will calc the coefficients of the polynomial from roots.
    // i.e., (x - root1) * (x - root2) * (x - root3) = x^3 - (root1 + root2 + root3)x^2 + (root1*root2 + root2*root3 + root3*root1)x - root1*root2*root3
    scaled_poly[0] = 1.0;
    scaled_poly[1] = -(scaled_roots[0] + scaled_roots[1] + scaled_roots[2]);
    scaled_poly[2] = scaled_roots[0] * scaled_roots[1] + scaled_roots[1] * scaled_roots[2] + scaled_roots[2] * scaled_roots[0];
    scaled_poly[3] = -scaled_roots[0] * scaled_roots[1] * scaled_roots[2];

    // Scale coefficients to enforce C(0) = 1.0
    float c0 = scaled_poly[3];
    for (int i = 0; i < degree; i++)
    {
        scaled_poly[i] /= c0;
    }

    chebyshev_coeff.resize(degree);
    // CAUTION:setup_chebyshev has "-" at the end
    for (int i = 0; i < degree; i++)
    {
        chebyshev_coeff[i] = -scaled_poly[i];
    }

    if (verbose)
    {
        cout << "Chebyshev polynomial coefficients: ";
        for (int i = 0; i < degree; i++)
        {
            cout << chebyshev_coeff[i] << " ";
        }
        cout << endl;
    }
}

void Smoother::chebyshev(int lv, Vec<float> &x, Vec<float> const &b)
{
    copy(levels.at(lv).residual, b);
    spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff);         // residual = b - A@x
    scal2(levels.at(lv).h, chebyshev_coeff.at(0), levels.at(lv).residual); // h = c0 * residual

    for (int i = 1; i < chebyshev_coeff.size(); ++i)
    {
        // h' = ci * residual + A@h
        copy(levels.at(lv).outh, levels.at(lv).residual);
        spmv(levels.at(lv).outh, 1, levels.at(lv).A, levels.at(lv).h, chebyshev_coeff.at(i), buff);

        // copy(levels.at(lv).h, levels.at(lv).outh);
        levels.at(lv).h.swap(levels.at(lv).outh);
    }

    axpy(x, 1, levels.at(lv).h); // x += h
}

void Smoother::set_smoother_niter(size_t const n)
{
    smoother_niter = n;
}

void Smoother::setup_weighted_jacobi()
{
    int nlvs = levels.size();

    if (use_radical_omega)
    {
        // old way:
        // use only the A0 omega for all, and set radical omega(estimate lambda_min as 0.1)
        // TODO: calculate omega for each level, and calculate lambda_min
        levels.at(0).jacobi_omega = calc_weighted_jacobi_omega(levels[0].A, true);
        cout << "omega: " << levels.at(0).jacobi_omega << endl;

        
        if (nlvs > 1)
            for (size_t lv = 1; lv < nlvs; lv++)
            {
                levels.at(lv).jacobi_omega = levels.at(0).jacobi_omega;
            }

        // // new way
        // for (size_t lv = 0; lv < nlvs; lv++)
        // {
        //     levels.at(lv).jacobi_omega = calc_weighted_jacobi_omega(levels[lv].A, true);
        // }
    }
    else
    {
        for (size_t lv = 0; lv < nlvs; lv++)
        {
            levels.at(lv).jacobi_omega = calc_weighted_jacobi_omega(levels[lv].A, false);
        }
    }
}

// FIXME: this has bugs, taking too long time
// https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csreigvsi
// calculate the most close to 0.1 eigen value of a symmetric matrix using the shift inverse method
float Smoother::calc_min_eig(CSR<float> &A, float mu0)
{
    cusparseMatDescr_t descrA = NULL;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    float tol = 1e-3;
    int maxite = 10;
    float *mu = NULL; // result eigen value
    Vec<float> x;     // result eigen vector
    x.resize(A.nrows);

    Vec<float> x0; // initial guess
    x0.resize(A.nrows);

    // // set initial guess as random
    // thrust::device_vector<float> x0(A.nrows);
    // thrust::transform(thrust::make_counting_iterator<int>(0),
    // thrust::make_counting_iterator<int>(A.nrows),
    // x0.begin(),
    // genRandomNumber());
    // float* x0_raw = thrust::raw_pointer_cast(x0.data());

    cusolverStatus_t t = cusolverSpScsreigvsi(cusolverH,
                                              A.nrows,
                                              A.numnonz,
                                              descrA,
                                              A.data.data(),
                                              A.indptr.data(),
                                              A.indices.data(),
                                              mu0,
                                              x0.data(),
                                              // thrust::raw_pointer_cast(x0.data()),
                                              maxite,
                                              tol,
                                              mu,
                                              x.data());
    CHECK_CUSOLVER(t);

    cout << "mu: " << *mu << endl;
    return *mu;
}

float Smoother::calc_weighted_jacobi_omega(CSR<float> &A, bool use_radical_omega)
{
    GpuTimer timer;
    timer.start();

    // calc Dinv@A
    // Vec<float> Dinv;
    Vec<float> data_new;
    Vec<float> diag_inv;
    // Dinv.resize(A.nrows);
    data_new.resize(A.data.size());
    diag_inv.resize(A.nrows);
    calc_diag_inv_kernel<<<(A.nrows + 255) / 256, 256>>>(diag_inv.data(), A.data.data(), A.indices.data(), A.indptr.data(), A.nrows);
    cudaDeviceSynchronize();

    scale_csr_by_row<<<(A.nrows + 255) / 256, 256>>>(data_new.data(), A.data.data(), A.indices.data(), A.indptr.data(), A.nrows, diag_inv.data());
    cudaDeviceSynchronize();
    LAUNCH_CHECK();

    CSR<float> DinvA;
    DinvA.assign(data_new.data(), A.numnonz, A.indices.data(), A.numnonz, A.indptr.data(), A.nrows + 1, A.nrows, A.ncols, A.numnonz);

    // TODO: calculate lambda_min
    float lambda_max = calc_max_eig(DinvA);
    float lambda_min;
    if (use_radical_omega)
    {
        cout << "use radical omega" << endl;
        lambda_min = 0.1;
        // lambda_min = calc_min_eig(DinvA);
    }
    else
    {
        lambda_min = lambda_max;
    }
    float jacobi_omega = 1.0 / (lambda_max + lambda_min);

    timer.stop();
    if (verbose)
        cout << "calc_weighted_jacobi_omega time: " << timer.elapsed() << " ms" << endl;
    return jacobi_omega;
}


void Smoother::jacobi(int lv, Vec<float> &x, Vec<float> const &b)
{
    Vec<float> x_old;
    x_old.resize(x.size());
    copy(x_old, x);
    auto jacobi_omega = levels.at(lv).jacobi_omega;
    for (int i = 0; i < smoother_niter; ++i)
    {
        weighted_jacobi_kernel<<<(levels.at(lv).A.nrows + 255) / 256, 256>>>(x.data(), x_old.data(), b.data(), levels.at(lv).A.data.data(), levels.at(lv).A.indices.data(), levels.at(lv).A.indptr.data(), levels.at(lv).A.nrows, jacobi_omega);
        x.swap(x_old);
    }
}

// use cusparse instead of hand-written kernel
void Smoother::jacobi_v2(int lv, Vec<float> &x, Vec<float> const &b)
{
    auto jacobi_omega = levels.at(lv).jacobi_omega;

    Vec<float> x_old;
    x_old.resize(x.size());
    copy(x_old, x);

    Vec<float> b1, b2;
    b1.resize(b.size());
    b2.resize(b.size());
    for (int i = 0; i < smoother_niter; ++i)
    {
        // x = omega * Dinv * (b - Aoff@x_old) + (1-omega)*x_old

        // 1. b1 = b-Aoff@x_old
        copy(b1, b);
        spmv(b1, -1, levels.at(lv).Aoff, x_old, 1, buff);

        // 2. b2 = omega*Dinv@b1
        spmv(b2, jacobi_omega, levels.at(lv).Dinv, b1, 0, buff);

        // 3. x = b2 + (1-omega)*x_old
        copy(x, x_old);
        axpy(x, 1 - jacobi_omega, b2);

        x.swap(x_old);
    }
}




// https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/amg_core/relaxation.h#L45
void gauss_seidel_serial(const int Ap[], const int Ap_size,
                  const int Aj[], const int Aj_size,
                  const float Ax[], const int Ax_size,
                        float  x[], const int  x_size,
                  const float  b[], const int  b_size,
                  const int row_start,
                  const int row_stop,
                  const int row_step)
{
    for(int i = row_start; i != row_stop; i += row_step) {
        int start = Ap[i];
        int end   = Ap[i+1];
        float rsum = 0;
        float diag = 0;

        for(int jj = start; jj < end; jj++){
            int j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (float) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}


void Smoother::gauss_seidel_cpu(int lv, Vec<float> &x, Vec<float> const &b)
{
    std::vector<float> x_host(x.size());
    std::vector<float> b_host(b.size());
    x.tohost(x_host);
    b.tohost(b_host);
    std::vector<float> data_host;
    std::vector<int> indices_host, indptr_host;
    levels.at(lv).A.tohost(data_host, indices_host, indptr_host);
    gauss_seidel_serial(
        indptr_host.data(), indptr_host.size(),
        indices_host.data(), indices_host.size(),
        data_host.data(), data_host.size(),
        x_host.data(), x_host.size(),
        b_host.data(), b_host.size(),
        0, levels.at(lv).A.nrows, 1);
    x.assign(x_host.data(), x_host.size());
}

// parallel gauss seidel
// https://erkaman.github.io/posts/gauss_seidel_graph_coloring.html
// https://gist.github.com/Erkaman/b34b3531e209a1db38e259ea53ff0be9#file-gauss_seidel_graph_coloring-cpp-L101
void Smoother::set_colors(const int *c, int n, int color_num_in, int lv)
{
    levels.at(lv).colors.resize(n);
    CHECK_CUDA(cudaMemcpy(levels.at(lv).colors.data(), c, n * sizeof(int), cudaMemcpyHostToDevice));
    levels.at(lv).color_num = color_num_in;
}

void Smoother::multi_color_gauss_seidel(int lv, Vec<float> &x, Vec<float> const &b)
{
    for (int color = 0; color < levels.at(lv).color_num; color++)
    {
        multi_color_gauss_seidel_kernel<<<(levels.at(lv).A.nrows + 255) / 256, 256>>>(x.data(), b.data(), levels.at(lv).A.data.data(), levels.at(lv).A.indices.data(), levels.at(lv).A.indptr.data(), levels.at(lv).A.nrows, levels.at(lv).colors.data(), color);
    }
}

void Smoother::smooth(int lv, Vec<float> &x, Vec<float> const &b)
{
    if (smoother_type == 1)
    {
        for (int i = 0; i < smoother_niter; i++)
            chebyshev(lv, x, b);
    }
    else if (smoother_type == 2)
    {
        // jacobi_cpu(lv, x, b);
        // jacobi(lv, x, b);
        jacobi_v2(lv, x, b);
    }
    else if (smoother_type == 3)
    {
        for (int i = 0; i < smoother_niter; i++)
            multi_color_gauss_seidel(lv, x, b);
    }
}

float Smoother::calc_max_eig(CSR<float>& A)
{
    return  computeMaxEigenvaluePowerMethodOptimized(A, 100);
}


} // namespace fastmg
