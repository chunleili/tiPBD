#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>
#include <numeric>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>

#include "kernels.cuh"
#include "cuda_utils.cuh"
#include "fastmg.h"
#include "fastfill.h"

using std::cout;
using std::endl;

#define USE_LESSMEM 1
#define VERBOSE 0

namespace fastmg{


float sum(std::vector<float> &v)
{
    return std::accumulate(v.begin(), v.end(), 0.0);
}

float avg(std::vector<float> &v)
{
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
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



/* -------------------------------------------------------------------------- */
/*                                   VCycle                                   */
/* -------------------------------------------------------------------------- */

    void VCycle::set_scale_RAP(float s, int lv)
    {
        levels.at(lv).scale_RAP = s;
        cout<<"Set scale_RAP: "<<levels.at(lv).scale_RAP<<"  at level "<<lv<<endl;
    }

    void  VCycle::setup_smoothers(int type) {
        if(VERBOSE)
            cout<<"\nSetting up smoothers..."<<endl;
        smoother_type = type;
        if(smoother_type == 1)
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


    void  VCycle::setup_chebyshev_cuda(CSR<float> &A) {
        float lower_bound=1.0/30.0;
        float upper_bound=1.1;
        float rho = computeMaxEigenvaluePowerMethodOptimized(A, 100);
        float a = rho * lower_bound;
        float b = rho * upper_bound;
        chebyshev_polynomial_coefficients(a, b);
        
        max_eig = rho;
        if (VERBOSE)
        {
            cout<<"max eigenvalue: "<<max_eig<<endl;
        }
    }


    void  VCycle::chebyshev_polynomial_coefficients(float a, float b)
    {
        int degree=3;
        const float PI = 3.14159265358979323846;

        if(a >= b || a <= 0)
            assert(false && "Invalid input for Chebyshev polynomial coefficients");

        // Chebyshev roots for the interval [-1,1]
        std::vector<float> std_roots(degree);
        for(int i=0; i<degree; i++)
        {
            std_roots[i] = std::cos(PI * (i + 0.5) / (float)degree);
        }

        // Chebyshev roots for the interval [a,b]
        std::vector<float> scaled_roots(degree);
        for(int i=0; i<degree; i++)
        {
            scaled_roots[i] = 0.5 * (b-a) * (1 + std_roots[i]) + a;
        }

        // Compute monic polynomial coefficients of polynomial with scaled roots
        std::vector<float> scaled_poly(4);
        // np.poly for 3 roots. This will calc the coefficients of the polynomial from roots.
        // i.e., (x - root1) * (x - root2) * (x - root3) = x^3 - (root1 + root2 + root3)x^2 + (root1*root2 + root2*root3 + root3*root1)x - root1*root2*root3
        scaled_poly[0] = 1.0;
        scaled_poly[1] = -(scaled_roots[0] + scaled_roots[1] + scaled_roots[2]);
        scaled_poly[2] = scaled_roots[0]*scaled_roots[1] + scaled_roots[1]*scaled_roots[2] + scaled_roots[2]*scaled_roots[0];
        scaled_poly[3] = -scaled_roots[0]*scaled_roots[1]*scaled_roots[2];

        // Scale coefficients to enforce C(0) = 1.0
        float c0 = scaled_poly[3];
        for(int i=0; i<degree; i++)
        {
            scaled_poly[i] /= c0; 
        }


        chebyshev_coeff.resize(degree);
        //CAUTION:setup_chebyshev has "-" at the end
        for(int i=0; i<degree; i++)
        {
            chebyshev_coeff[i] = -scaled_poly[i];
        }

        if(VERBOSE)
        {
            cout<<"Chebyshev polynomial coefficients: ";
            for(int i=0; i<degree; i++)
            {
                cout<<chebyshev_coeff[i]<<" ";
            }
            cout<<endl;
        }
    }


    float  VCycle::calc_residual_norm(Vec<float> const &b, Vec<float> const &x, CSR<float> const &A) {
        float rnorm = 0.0;
        Vec<float> r;
        r.resize(b.size());
        copy(r, b);
        spmv(r, -1, A, x, 1, buff);
        rnorm = vnorm(r);
        return rnorm;
    }


    void  VCycle::setup(size_t numlvs) {
        if (levels.size() < numlvs) {
            levels.resize(numlvs);
        }
        nlvs = numlvs;
    }


    void  VCycle::set_P(size_t lv, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        levels.at(lv).P.assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    }

    void  VCycle::set_A0(float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        levels.at(0).A.assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    }


    int  VCycle::get_nnz(int lv) {
        return levels.at(lv).A.numnonz;
    }

    int  VCycle::get_nrows(int lv) {
        return levels.at(lv).A.nrows;
    }

    // only update the data of A0
    void  VCycle::update_A0(float const *datap) {
        CHECK_CUDA(cudaMemcpy(levels.at(0).A.data.data(), datap, levels.at(0).A.data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void  VCycle::set_A0_from_fastFill(FastFillBase *ff)
    {
        if (levels.size() < 1) {
            levels.resize(1);
        }

        levels.at(0).A.numnonz = ( ff->num_nonz);
        levels.at(0).A.nrows = ( ff->nrows);

        //FIXME: As in python code, we need copy A, why?
        levels.at(0).A.data.resize(ff->A.data.size());
        CHECK_CUDA(cudaMemcpy(levels.at(0).A.data.data(), (ff->A).data.data(), (ff->A).data.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        levels.at(0).A.indices.resize(ff->A.indices.size());
        CHECK_CUDA(cudaMemcpy(levels.at(0).A.indices.data(), (ff->A).indices.data(), (ff->A).indices.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        levels.at(0).A.indptr.resize(ff->A.indptr.size());
        CHECK_CUDA(cudaMemcpy(levels.at(0).A.indptr.data(), (ff->A).indptr.data(), (ff->A).indptr.size() * sizeof(int), cudaMemcpyDeviceToDevice));
    }


    void  VCycle::chebyshev(int lv, Vec<float> &x, Vec<float> const &b) {
        copy(levels.at(lv).residual, b);
        spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x
        scal2(levels.at(lv).h, chebyshev_coeff.at(0), levels.at(lv).residual); // h = c0 * residual


        for (int i = 1; i < chebyshev_coeff.size(); ++i) {
            // h' = ci * residual + A@h
            copy(levels.at(lv).outh, levels.at(lv).residual);
            spmv(levels.at(lv).outh, 1, levels.at(lv).A, levels.at(lv).h, chebyshev_coeff.at(i), buff);

            // copy(levels.at(lv).h, levels.at(lv).outh);
            levels.at(lv).h.swap(levels.at(lv).outh);
        }

        axpy(x, 1, levels.at(lv).h); // x += h
    }


    void  VCycle::set_smoother_niter(size_t const n) {
        smoother_niter = n;
    }


    void  VCycle::setup_weighted_jacobi() {
        if(use_radical_omega)
        {
            // old way:
            // use only the A0 omega for all, and set radical omega(estimate lambda_min as 0.1)
            // TODO: calculate omega for each level, and calculate lambda_min
            levels.at(0).jacobi_omega = calc_weighted_jacobi_omega(levels[0].A, true);
            cout<<"omega: "<<levels.at(0).jacobi_omega<<endl;
            if(nlvs>1)
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
    float  VCycle::calc_min_eig(CSR<float> &A, float mu0) {
        cusparseMatDescr_t descrA = NULL;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
        CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 

        float tol=1e-3;
        int maxite=10;
        float* mu=NULL; //result eigen value
        Vec<float> x;//result eigen vector
        x.resize(A.nrows);

        Vec<float> x0; //initial guess
        x0.resize(A.nrows);

        // // set initial guess as random
        // thrust::device_vector<float> x0(A.nrows);
        // thrust::transform(thrust::make_counting_iterator<int>(0),
        // thrust::make_counting_iterator<int>(A.nrows),
        // x0.begin(),
        // genRandomNumber());
        // float* x0_raw = thrust::raw_pointer_cast(x0.data());

        cusolverStatus_t t= cusolverSpScsreigvsi(cusolverH,
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
            
        cout<<"mu: "<<*mu<<endl;
        return *mu;
    }


    float  VCycle::calc_weighted_jacobi_omega(CSR<float>&A, bool use_radical_omega) {
        GpuTimer timer;
        timer.start();

        // calc Dinv@A
        // Vec<float> Dinv;
        Vec<float> data_new;
        Vec<float> diag_inv;
        // Dinv.resize(A.nrows);
        data_new.resize(A.data.size());
        diag_inv.resize(A.nrows);
        calc_diag_inv_kernel<<<(A.nrows + 255) / 256, 256>>>(diag_inv.data(),A.data.data(), A.indices.data(), A.indptr.data(), A.nrows);
        cudaDeviceSynchronize();

        scale_csr_by_row<<<(A.nrows + 255) / 256, 256>>>(data_new.data(), A.data.data(), A.indices.data(), A.indptr.data(), A.nrows, diag_inv.data());
        cudaDeviceSynchronize();
        LAUNCH_CHECK();

        CSR<float> DinvA;
        DinvA.assign(data_new.data(), A.numnonz, A.indices.data(), A.numnonz, A.indptr.data(), A.nrows+1, A.nrows, A.ncols, A.numnonz);


        // TODO: calculate lambda_min
        float lambda_max = calc_max_eig(DinvA);
        float lambda_min;
        if (use_radical_omega)
        {
            cout<<"use radical omega"<<endl;
            lambda_min = 0.1;
            // lambda_min = calc_min_eig(DinvA);
        }
        else 
        {
            lambda_min = lambda_max;
        }
        float jacobi_omega = 1.0 / (lambda_max+lambda_min);
 
        timer.stop();
        if(VERBOSE)
            cout<<"calc_weighted_jacobi_omega time: "<<timer.elapsed()<<" ms"<<endl;
        return jacobi_omega;
    }


    void  VCycle::get_Aoff_and_Dinv(CSR<float> &A, CSR<float> &Dinv, CSR<float> &Aoff)
    {
        int n = A.nrows;
        // get diagonal inverse of A, fill into a vector
        Vec<float> d_diag_inv;
        d_diag_inv.resize(n);
        calc_diag_inv_kernel<<<(n + 255) / 256, 256>>>(d_diag_inv.data(),A.data.data(), A.indices.data(), A.indptr.data(), n);
        cudaDeviceSynchronize();
        LAUNCH_CHECK();


        // fill diag to a CSR matrix Dinv
        std::vector<int> seqence(n);
        for(int i=0; i<n; i++)
            seqence[i] = i;
        // copy d_diag_inv to host
        std::vector<float> h_diag_inv(n);
        CHECK_CUDA(cudaMemcpy(h_diag_inv.data(), d_diag_inv.data(), n*sizeof(float), cudaMemcpyDeviceToHost));
        Dinv.assign_v2(h_diag_inv.data(), seqence.data(), seqence.data(), n, n, n);
        cudaDeviceSynchronize();
        LAUNCH_CHECK();


        Aoff.resize(n,n,A.numnonz);
        CHECK_CUDA(cudaMemcpy(Aoff.data.data(), A.data.data(), A.numnonz*sizeof(float), cudaMemcpyDeviceToDevice));
        Aoff.assign(Aoff.data.data(), A.numnonz, A.indices.data(), A.numnonz, A.indptr.data(), n+1, n, n, A.numnonz);
        // get Aoff by set diagonal of A to 0
        get_Aoff_kernel<<<(A.numnonz + 255) / 256, 256>>>(Aoff.data.data(), A.indices.data(), A.indptr.data(), n);
        cudaDeviceSynchronize();
        LAUNCH_CHECK();
    }


    void  VCycle::jacobi(int lv, Vec<float> &x, Vec<float> const &b) {
        Vec<float> x_old;
        x_old.resize(x.size());
        copy(x_old, x);
        auto jacobi_omega = levels.at(lv).jacobi_omega;
        for (int i = 0; i < smoother_niter; ++i) {
            weighted_jacobi_kernel<<<(levels.at(lv).A.nrows + 255) / 256, 256>>>(x.data(), x_old.data(), b.data(), levels.at(lv).A.data.data(), levels.at(lv).A.indices.data(), levels.at(lv).A.indptr.data(), levels.at(lv).A.nrows, jacobi_omega);
            x.swap(x_old);
        }
    }

    // use cusparse instead of hand-written kernel
    void  VCycle::jacobi_v2(int lv, Vec<float> &x, Vec<float> const &b) {
        auto jacobi_omega = levels.at(lv).jacobi_omega;

        Vec<float> x_old;
        x_old.resize(x.size());
        copy(x_old, x);

        Vec<float> b1,b2;
        b1.resize(b.size());
        b2.resize(b.size());
        for (int i = 0; i < smoother_niter; ++i) {
            //x = omega * Dinv * (b - Aoff@x_old) + (1-omega)*x_old

            // 1. b1 = b-Aoff@x_old
            copy(b1, b);
            spmv(b1, -1, levels.at(lv).Aoff, x_old, 1, buff);

            // 2. b2 = omega*Dinv@b1
            spmv(b2, jacobi_omega, levels.at(lv).Dinv, b1, 0, buff);

            // 3. x = b2 + (1-omega)*x_old
            copy(x, x_old);
            axpy(x, 1-jacobi_omega, b2);

            x.swap(x_old);
        }   
    }


    void  VCycle::gauss_seidel_cpu(int lv, Vec<float> &x, Vec<float> const &b) {
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
    void  VCycle::set_colors(const int* c, int n, int color_num_in, int lv) {
        levels.at(lv).colors.resize(n);
        CHECK_CUDA(cudaMemcpy(levels.at(lv).colors.data(), c, n*sizeof(int), cudaMemcpyHostToDevice));
        levels.at(lv).color_num = color_num_in;

    }

    void  VCycle::multi_color_gauss_seidel(int lv, Vec<float> &x, Vec<float> const &b) {
        for(int color=0; color<levels.at(lv).color_num; color++)
        {
            multi_color_gauss_seidel_kernel<<<(levels.at(lv).A.nrows + 255) / 256, 256>>>(x.data(), b.data(), levels.at(lv).A.data.data(), levels.at(lv).A.indices.data(), levels.at(lv).A.indptr.data(), levels.at(lv).A.nrows, levels.at(lv).colors.data(), color);
        }

    }


    void  VCycle::_smooth(int lv, Vec<float> &x, Vec<float> const &b) {
        if(smoother_type == 1)
        {
            for(int i=0; i<smoother_niter; i++)
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
            for(int i=0; i<smoother_niter; i++)
                multi_color_gauss_seidel(lv,x,b);
        }
    }


    float  VCycle::calc_residual(int lv, CSR<float> const &A, Vec<float> &x, Vec<float> const &b) {
        copy(r, b);
        spmv(r, -1, A, x, 1, buff); // residual = b - A@x
        return vnorm(r);
    }


    void  VCycle::vcycle_down() {
        for (int lv = 0; lv < nlvs-1; ++lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : z;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : r;

            _smooth(lv, x, b);

            copy(levels.at(lv).residual, b);
            spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x

            levels.at(lv).b.resize(levels.at(lv).R.nrows);
            spmv(levels.at(lv).b, 1, levels.at(lv).R, levels.at(lv).residual, 0, buff); // coarse_b = R@residual

            levels.at(lv).x.resize(levels.at(lv).b.size());
            zero(levels.at(lv).x);
        }
    }

    void  VCycle::vcycle_up() {
        for (int lv = nlvs-2; lv >= 0; --lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : z;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : r;
            spmv(x, 1, levels.at(lv).P, levels.at(lv).x, 1, buff); // x += P@coarse_x
            _smooth(lv, x, b);
        }
    }

    void  VCycle::vcycle() {
        vcycle_down();
        coarse_solve();
        vcycle_up();
    }


    void  VCycle::coarse_solve() {
        auto const &A = levels.at(nlvs - 1).A;
        auto &x = levels.at(nlvs - 2).x;
        auto &b = levels.at(nlvs - 2).b;
        if (coarse_solver_type==0)
        {
            spsolve(x, A, b);
        }
        else if (coarse_solver_type==1)
        {
            _smooth(nlvs-1, x, b);
        }
    }

    void  VCycle::set_outer_x(float const *x, size_t n) {
        outer_x.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
        copy(x_new, outer_x);
    }

    void  VCycle::set_outer_b(float const *b, size_t n) {
        outer_b.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_b.data(), b, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    float  VCycle::init_cg_iter0(float *residuals) {
        float bnrm2 = vnorm(outer_b);
        // r = b - A@(x)
        copy(r, outer_b);
        spmv(outer_b, -1, levels.at(0).A, outer_x, 1, buff);
        float normr = vnorm(r);
        residuals[0] = normr;
        return bnrm2;
    }

    void  VCycle::do_cg_itern(float *residuals, size_t iteration) {
        float rho_cur = vdot(r, z);
        if (iteration > 0) {
            float beta = rho_cur / save_rho_prev;
            // p *= beta
            // p += z
            scal(save_p, beta);
            axpy(save_p, 1, z);
        } else {
            // p = move(z)
            save_p.swap(z);
        }
        // q = A@(p)
        save_q.resize(levels.at(0).A.nrows);
        spmv(save_q, 1, levels.at(0).A, save_p, 0, buff);
        save_rho_prev = rho_cur;
        float alpha = rho_cur / vdot(save_p, save_q);
        // x += alpha*p
        axpy(x_new, alpha, save_p);
        // r -= alpha*q
        axpy(r, -alpha, save_q);
        float normr = vnorm(r);
        residuals[iteration + 1] = normr;
    }

    void  VCycle::compute_RAP(size_t lv) {
            CSR<float> &A = levels.at(lv).A;
            CSR<float> &R = levels.at(lv).R;
            CSR<float> &P = levels.at(lv).P;
            CSR<float> AP;
            CSR<float> &RAP = levels.at(lv+1).A;
            R.resize(P.ncols, P.nrows, P.numnonz);
            transpose(P, R);            
            spgemm(A, P, AP) ;
            spgemm(R, AP, RAP);

            float s = levels.at(lv).scale_RAP;
            if (s!=0.0){
                // scale RAP by a scalar
                cout<<"scaling RAP by "<<s<<" at lv "<<lv<<endl;
                scal(RAP.data, s);
            }
    }

    void  VCycle::fetch_A_data(float *data) {
        CSR<float> &A = levels.at(0).A;
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // In python end, before you call fetch A, you should call get_nnz and get_matsize first to determine the size of the csr matrix. 
    void  VCycle::fetch_A(size_t lv, float *data, int *indices, int *indptr) {
        CSR<float> &A = levels.at(lv).A;
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    void  VCycle::set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol_, size_t maxiter_)
    {
        set_outer_x(x, nx);
        set_outer_b(b, nb);
        rtol = rtol_;
        maxiter = maxiter_;
        residuals.resize(maxiter+1);
    }

    float  VCycle::calc_max_eig(CSR<float>& A)
    {
        return  computeMaxEigenvaluePowerMethodOptimized(A, 100);
    }

    size_t  VCycle::get_data(float* x_out, float* r_out)
    {
        CHECK_CUDA(cudaMemcpy(x_out, x_new.data(), x_new.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(residuals.begin(), residuals.end(), r_out);
        return niter;
    }


    void  VCycle::presolve()
    {
        // TODO: move fillA from python-end to here as well in the future refactoring
        for(int lv=0; lv<nlvs; lv++)
        {
            // for jacobi_v2 (use cusparse etc.)
            if(smoother_type == 2)
            {
                get_Aoff_and_Dinv(levels.at(lv).A, levels.at(lv).Dinv, levels.at(lv).Aoff);
            }

        }
        for (size_t lv = 0; lv < nlvs-1; lv++)
        {
            compute_RAP(lv);
        }
        
    }

    void  VCycle::solve()
    {
        presolve();
        float bnrm2 = init_cg_iter0(residuals.data());
        float atol = bnrm2 * rtol;
        for (size_t iter=0; iter<maxiter; iter++)
        {   
            if (residuals[iter] < atol)
            {
                niter = iter;
                break;
            }
            copy(z, outer_x);
            vcycle();
            do_cg_itern(residuals.data(), iter); 
            niter = iter;
        }
    }

    void  VCycle::solve_only_jacobi()
    {
        timer1.start();
        get_Aoff_and_Dinv(levels.at(0).A, levels.at(0).Dinv, levels.at(0).Aoff);
        for (size_t iter=0; iter<maxiter; iter++)
            jacobi_v2(0, outer_x, outer_b);
        copy(x_new, outer_x);
        
        timer1.stop();
        elapsed1.push_back(timer1.elapsed());
        // if (verbose)
            cout<<" only iterative time: "<<(elapsed1[0])<<" ms"<<endl;
        elapsed1.clear();
    }

    void  VCycle::solve_only_directsolver()
    {
        timer1.start();

        spsolve(outer_x, levels.at(0).A, outer_b);
        copy(x_new, outer_x);
        
        timer1.stop();
        elapsed1.push_back(timer1.elapsed());
        // if (verbose)
            cout<<" only direct time: "<<(elapsed1[0])<<" ms"<<endl;
        elapsed1.clear();
    }

    void  VCycle::solve_only_smoother()
    {
        timer1.start();
        presolve();
        float bnrm2 = init_cg_iter0(residuals.data());
        float atol = bnrm2 * rtol;
        for (size_t iter=0; iter<maxiter; iter++)
        {   
            _smooth(0, outer_x, outer_b);
            auto r = calc_residual_norm(outer_b, outer_x, levels.at(0).A);
            residuals[iter] = r;
            if (residuals[iter] < atol)
            {
                niter = iter;
                break;
            }
            niter = iter;
        }
        copy(x_new, outer_x);

        timer1.stop();
        elapsed1.push_back(timer1.elapsed());
        cout<<elapsed1.size()<<" only smoother time: "<<(elapsed1[0])<<" ms"<<endl;
        elapsed1.clear();

    }




VCycle *fastmg = nullptr;


#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


extern "C" DLLEXPORT void fastmg_set_A0_from_fastFillCloth() {
    fastmg->set_A0_from_fastFill(fastFillCloth);
}

extern "C" DLLEXPORT void fastmg_set_A0_from_fastFillSoft() {
    fastmg->set_A0_from_fastFill(fastFillSoft);
}



extern "C" DLLEXPORT void fastmg_new() {
    if (!fastmg)
        fastmg = new VCycle{};
}

extern "C" DLLEXPORT void fastmg_setup_nl(size_t numlvs) {
    fastmg->setup(numlvs);
}


extern "C" DLLEXPORT void fastmg_RAP(size_t lv) {
    fastmg->compute_RAP(lv);
}


extern "C" DLLEXPORT int fastmg_get_nnz(size_t lv) {
    int nnz = fastmg->get_nnz(lv);
    std::cout<<"nnz: "<<nnz<<std::endl;
    return nnz;
}

extern "C" DLLEXPORT int fastmg_get_matsize(size_t lv) {
    int n = fastmg->get_nrows(lv);
    std::cout<<"matsize: "<<n<<std::endl;
    return n;
}

extern "C" DLLEXPORT void fastmg_fetch_A(size_t lv, float* data, int* indices, int* indptr) {
    fastmg->fetch_A(lv, data, indices, indptr);
}

extern "C" DLLEXPORT void fastmg_fetch_A_data(float* data) {
    fastmg->fetch_A_data(data);
}

extern "C" DLLEXPORT void fastmg_solve() {
    fastmg->solve();
}

extern "C" DLLEXPORT void fastmg_set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol, size_t maxiter) {
    fastmg->set_data(x, nx, b, nb, rtol, maxiter);
}

extern "C" DLLEXPORT size_t fastmg_get_data(float *x, float *r) {
    size_t niter = fastmg->get_data(x, r);
    return niter;
}

extern "C" DLLEXPORT void fastmg_set_A0(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastmg->set_A0(data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}

// only update the data of A0
extern "C" DLLEXPORT void fastmg_update_A0(const float* data_in)
{
    fastmg->update_A0(data_in);
}

extern "C" DLLEXPORT void fastmg_set_P(int lv, float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastmg->set_P(lv, data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}


extern "C" DLLEXPORT void fastmg_setup_smoothers(int type) {
    fastmg->setup_smoothers(type);
}


extern "C" DLLEXPORT void fastmg_set_smoother_niter(const size_t niter) {
    fastmg->set_smoother_niter(niter);
}



extern "C" DLLEXPORT void fastmg_scale_RAP(float s, int lv) {
    fastmg->set_scale_RAP(s, lv);
}

extern "C" DLLEXPORT void fastmg_set_colors(const int *c, int n, int color_num, int lv) {
    fastmg->set_colors(c, n, color_num, lv);
}


extern "C" DLLEXPORT void fastmg_solve_only_smoother() {
    fastmg->solve_only_smoother();
}


extern "C" DLLEXPORT void fastmg_solve_only_jacobi() {
    fastmg->solve_only_jacobi();
}

extern "C" DLLEXPORT void fastmg_solve_only_directsolver() {
    fastmg->solve_only_directsolver();
}

extern "C" DLLEXPORT void fastmg_set_coarse_solver_type(int t) {
    fastmg->coarse_solver_type = t;
}


extern "C" DLLEXPORT void fastmg_use_radical_omega(int flag) {
    fastmg->use_radical_omega = bool(flag);
}


} // namespace
