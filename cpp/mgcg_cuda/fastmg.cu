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
#include "cusparse_wrappers.h"
#include "mglevel.h"
#include "smoother.h"
#include "fastmg.h"
#include "fastfill.h"
#include "vcycle.h"
#include "Vec.h"
#include "CSR.h"

using std::cout;
using std::endl;

#define USE_LESSMEM 1

namespace fastmg{


float sum(std::vector<float> &v)
{
    return std::accumulate(v.begin(), v.end(), 0.0);
}

float avg(std::vector<float> &v)
{
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}



/* -------------------------------------------------------------------------- */
/*                                   VCycle                                   */
/* -------------------------------------------------------------------------- */

    void FastMG::set_scale_RAP(float s, int lv)
    {
        levels.at(lv).scale_RAP = s;
        cout<<"Set scale_RAP: "<<levels.at(lv).scale_RAP<<"  at level "<<lv<<endl;
    }


    void  FastMG::create_levels(size_t numlvs) {
        if (levels.size() < numlvs) {
            levels.resize(numlvs);
        }
        nlvs = numlvs;

        smoother = std::make_shared<Smoother>(levels);
        vcycle = std::make_unique<VCycle>(levels, smoother,z,r,buff);

    }


    void  FastMG::set_P(size_t lv, float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        levels.at(lv).P.assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    }

    void  FastMG::set_A0(float const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        levels.at(0).A.assign(datap, ndat, indicesp, nind, indptrp, nptr, rows, cols, nnz);
    }


    int  FastMG::get_nnz(int lv) {
        return levels.at(lv).A.numnonz;
    }

    int  FastMG::get_nrows(int lv) {
        return levels.at(lv).A.nrows;
    }

    // only update the data of A0
    void  FastMG::update_A0(float const *datap) {
        CHECK_CUDA(cudaMemcpy(levels.at(0).A.data.data(), datap, levels.at(0).A.data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void  FastMG::set_A0_from_fastFill(FastFillBase *ff)
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



    float  FastMG::calc_residual(CSR<float> const &A, Vec<float> &x, Vec<float> const &b, Vec<float> &r) {
        copy(r, b);
        spmv(r, -1, A, x, 1, buff); // residual = b - A@x
        return vnorm(r);
    }



    void  FastMG::set_outer_x(float const *x, size_t n) {
        outer_x.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_x.data(), x, n * sizeof(float), cudaMemcpyHostToDevice));
        copy(x_new, outer_x);
    }

    void  FastMG::set_outer_b(float const *b, size_t n) {
        outer_b.resize(n);
        CHECK_CUDA(cudaMemcpy(outer_b.data(), b, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    float  FastMG::init_cg_iter0(float *residuals) {
        float bnrm2 = vnorm(outer_b);
        // r = b - A@(x)
        copy(r, outer_b);
        spmv(outer_b, -1, levels.at(0).A, outer_x, 1, buff);
        float normr = vnorm(r);
        residuals[0] = normr;
        return bnrm2;
    }

    void  FastMG::do_cg_itern(float *residuals, size_t iteration) {
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

    void  FastMG::compute_RAP(size_t lv) {
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

    void  FastMG::fetch_A_data(float *data) {
        CSR<float> &A = levels.at(0).A;
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // In python end, before you call fetch A, you should call get_nnz and get_matsize first to determine the size of the csr matrix. 
    void  FastMG::fetch_A(size_t lv, float *data, int *indices, int *indptr) {
        CSR<float> &A = levels.at(lv).A;
        CHECK_CUDA(cudaMemcpy(data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    void  FastMG::set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol_, size_t maxiter_)
    {
        set_outer_x(x, nx);
        set_outer_b(b, nb);
        rtol = rtol_;
        maxiter = maxiter_;
        residuals.resize(maxiter+1);
    }



    size_t  FastMG::get_data(float* x_out, float* r_out)
    {
        CHECK_CUDA(cudaMemcpy(x_out, x_new.data(), x_new.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(residuals.begin(), residuals.end(), r_out);
        return niter;
    }


    void  FastMG::presolve()
    {
        // TODO: move fillA from python-end to here as well in the future refactoring
        for(int lv=0; lv<nlvs; lv++)
        {
            // for jacobi_v2 (use cusparse etc.)
            if(smoother->smoother_type == 2)
            {
                get_Aoff_and_Dinv(levels.at(lv).A, levels.at(lv).Dinv, levels.at(lv).Aoff);
            }
        }
        for (size_t lv = 0; lv < nlvs-1; lv++)
        {
            compute_RAP(lv);
        }
        
    }

    void  FastMG::solve()
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
            vcycle -> run();
            do_cg_itern(residuals.data(), iter); 
            niter = iter;
        }
    }

    void  FastMG::solve_only_jacobi()
    {
        timer1.start();
        get_Aoff_and_Dinv(levels.at(0).A, levels.at(0).Dinv, levels.at(0).Aoff);
        for (size_t iter=0; iter<maxiter; iter++)
            smoother->jacobi_v2(0, outer_x, outer_b);
        copy(x_new, outer_x);
        
        timer1.stop();
        elapsed1.push_back(timer1.elapsed());
        // if (verbose)
            cout<<" only iterative time: "<<(elapsed1[0])<<" ms"<<endl;
        elapsed1.clear();
    }

    void  FastMG::solve_only_directsolver()
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

    void  FastMG::solve_only_smoother()
    {
        timer1.start();
        presolve();
        float bnrm2 = init_cg_iter0(residuals.data());
        float atol = bnrm2 * rtol;
        for (size_t iter=0; iter<maxiter; iter++)
        {   
            smoother->smooth(0, outer_x, outer_b);
            auto rnorm = calc_residual(levels.at(0).A, outer_x, outer_b, r);
            residuals[iter] = rnorm;
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


void FastMG::get_Aoff_and_Dinv(CSR<float> &A, CSR<float> &Dinv, CSR<float> &Aoff)
{
    int n = A.nrows;
    // get diagonal inverse of A, fill into a vector
    Vec<float> d_diag_inv;
    d_diag_inv.resize(n);
    calc_diag_inv_kernel<<<(n + 255) / 256, 256>>>(d_diag_inv.data(), A.data.data(), A.indices.data(), A.indptr.data(), n);
    cudaDeviceSynchronize();
    LAUNCH_CHECK();

    // fill diag to a CSR matrix Dinv
    std::vector<int> seqence(n);
    for (int i = 0; i < n; i++)
        seqence[i] = i;
    // copy d_diag_inv to host
    std::vector<float> h_diag_inv(n);
    CHECK_CUDA(cudaMemcpy(h_diag_inv.data(), d_diag_inv.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
    Dinv.assign_v2(h_diag_inv.data(), seqence.data(), seqence.data(), n, n, n);
    cudaDeviceSynchronize();
    LAUNCH_CHECK();

    Aoff.resize(n, n, A.numnonz);
    CHECK_CUDA(cudaMemcpy(Aoff.data.data(), A.data.data(), A.numnonz * sizeof(float), cudaMemcpyDeviceToDevice));
    Aoff.assign(Aoff.data.data(), A.numnonz, A.indices.data(), A.numnonz, A.indptr.data(), n + 1, n, n, A.numnonz);
    // get Aoff by set diagonal of A to 0
    get_Aoff_kernel<<<(A.numnonz + 255) / 256, 256>>>(Aoff.data.data(), A.indices.data(), A.indptr.data(), n);
    cudaDeviceSynchronize();
    LAUNCH_CHECK();
}


FastMG *fastmg = nullptr;


#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


extern "C" DLLEXPORT void fastmg_setup_smoothers(int type) {
    fastmg->smoother->setup_smoothers(type);
}


extern "C" DLLEXPORT void fastmg_set_smoother_niter(const size_t niter) {
    fastmg->smoother->set_smoother_niter(niter);
}

extern "C" DLLEXPORT void fastmg_set_colors(const int *c, int n, int color_num, int lv) {
    fastmg->smoother->set_colors(c, n, color_num, lv);
}

extern "C" DLLEXPORT void fastmg_use_radical_omega(int flag) {
    fastmg->smoother->use_radical_omega = bool(flag);
}


extern "C" DLLEXPORT void fastmg_set_coarse_solver_type(int t) {
    fastmg->vcycle->coarse_solver_type = t;
}


extern "C" DLLEXPORT void fastmg_set_A0_from_fastFillCloth() {
    fastmg->set_A0_from_fastFill(fastFillCloth);
}

extern "C" DLLEXPORT void fastmg_set_A0_from_fastFillSoft() {
    fastmg->set_A0_from_fastFill(fastFillSoft);
}



extern "C" DLLEXPORT void fastmg_new() {
    if (!fastmg)
        fastmg = new FastMG{};
}

extern "C" DLLEXPORT void fastmg_setup_nl(size_t numlvs) {
    fastmg->create_levels(numlvs);
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



extern "C" DLLEXPORT void fastmg_scale_RAP(float s, int lv) {
    fastmg->set_scale_RAP(s, lv);
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







} // namespace
