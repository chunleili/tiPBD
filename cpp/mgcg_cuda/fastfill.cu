#include "fastmg.h"
#include "fastfill.h"
#include "kernels.cuh"

namespace fastmg {
/* -------------------------------------------------------------------------- */
/*                                FastFillCloth                               */
/* -------------------------------------------------------------------------- */

void FastFillCloth::fetch_A_data(float *data_in) {
    CHECK_CUDA(cudaMemcpy(data_in, A.data.data(), sizeof(float) * A.numnonz, cudaMemcpyDeviceToHost));
}

void FastFillCloth::set_data_v2(int* edges_in, int NE_in, float* inv_mass_in, int NV_in, float* pos_in, float alpha_in)
{
    NE = NE_in;
    NV = NV_in;
    nrows = NE;
    ncols = NE;

    d_edges.assign(edges_in, NE*2);
    d_inv_mass.assign(inv_mass_in, NV);
    d_pos.assign(pos_in, NV*3);

    alpha = alpha_in;
}

void FastFillCloth::update_pos_py2cu(float* pos_in)
{
    d_pos.assign(pos_in, NV*3);
}


void FastFillCloth::init_from_python_cache_v2(
    int *adjacent_edge_in,
    int *num_adjacent_edge_in,
    int *adjacent_edge_abc_in,
    int num_nonz_in,
    float *spmat_data_in,
    int *spmat_indices_in,
    int *spmat_indptr_in,
    int *spmat_ii_in,
    int *spmat_jj_in,
    int NE_in,
    int NV_in)
{
    NE = NE_in;
    NV = NV_in;
    num_nonz = num_nonz_in;

    printf("Copying A, ii, jj\n");
    A.assign(spmat_data_in, num_nonz, spmat_indices_in, num_nonz, spmat_indptr_in, NE+1, NE, NE, num_nonz);
    d_ii.assign(spmat_ii_in, num_nonz);
    d_jj.assign(spmat_jj_in, num_nonz);
    cout<<"Finish."<<endl;

    printf("Copying adj\n");
    d_num_adjacent_edge.assign(num_adjacent_edge_in, NE);
    d_adjacent_edge_abc.resize(NE*60);
    CHECK_CUDA(cudaMemcpy(d_adjacent_edge_abc.data(), adjacent_edge_abc_in, sizeof(int) * NE * 60, cudaMemcpyHostToDevice));
    cout<<"Finish."<<endl;
}


void FastFillCloth::run(float* pos_in)
{
    update_pos_py2cu(pos_in);
    fill_A_CSR_gpu();
}


void FastFillCloth::fill_A_CSR_gpu()
{
    fill_A_CSR_cloth_kernel<<<num_nonz / 256 + 1, 256>>>(A.data.data(),
                                                A.indptr.data(),
                                                A.indices.data(),
                                                d_ii.data(),
                                                d_jj.data(),
                                                d_adjacent_edge_abc.data(),
                                                d_num_adjacent_edge.data(),
                                                num_nonz,
                                                d_inv_mass.data(),
                                                alpha,
                                                NV,
                                                NE,
                                                d_edges.data(),
                                                d_pos.data());
    cudaDeviceSynchronize();
    LAUNCH_CHECK();
}




/* -------------------------------------------------------------------------- */
/*                                FastFillSoft                                */
/* -------------------------------------------------------------------------- */


void FastFillSoft::fetch_A_data(float *data_in) {
    CHECK_CUDA(cudaMemcpy(data_in, A.data.data(), sizeof(float) * A.numnonz, cudaMemcpyDeviceToHost));
}

void FastFillSoft::set_data_v2(int* tet_in, int NT_in, float* inv_mass_in, int NV_in, float* pos_in, float* alpha_tilde_in)
{
    NT = NT_in;
    NV = NV_in;
    nrows = NT;
    ncols = NT;
    d_alpha_tilde.assign(alpha_tilde_in, NT);
    d_inv_mass.assign(inv_mass_in, NV);
    d_pos.assign(pos_in, NV*3);
    d_tet.assign(tet_in, NT*4);
}

void FastFillSoft::update_pos_and_gradC(float* pos_in, float* gradC_in)
{
    d_pos.assign(pos_in, NV*3);
    d_gradC.assign(gradC_in, NT*4*3);
}



void FastFillSoft::init_from_python_cache_lessmem(
    const int NT_in,
    const int MAX_ADJ_in,
    const float* data_in,
    const int* indices_in,
    const int* indptr_in,
    const int* ii_in,
    const int num_nonz_in
    )
{
    NT = NT_in;
    MAX_ADJ = MAX_ADJ_in;

    num_nonz = num_nonz_in;
    ncols = NT;
    nrows = NT;
    A.assign_v2(data_in, indices_in, indptr_in, NT, NT, num_nonz);
    d_ii.assign(ii_in, num_nonz_in);

    cout<<"Finish load python cache to cuda."<<endl;
}


void FastFillSoft::run(float* pos_in, float* gradC_in)
{
    update_pos_and_gradC(pos_in, gradC_in);
    fill_A_CSR_gpu();
}


void FastFillSoft::fill_A_CSR_gpu()
{
    
    fill_A_CSR_soft_lessmem_kernel<<<num_nonz / 256 + 1, 256>>>(
            A.data.data(),
            A.indptr.data(),
            A.indices.data(), //jj is the same as indices
            d_ii.data(),
            num_nonz,
            d_inv_mass.data(),
            d_alpha_tilde.data(),
            NV,
            NT,
            MAX_ADJ,
            d_tet.data(),
            d_pos.data(),
            d_gradC.data()
    );

    cudaDeviceSynchronize();
    LAUNCH_CHECK();
}


FastFillCloth *fastFillCloth = nullptr;
FastFillSoft *fastFillSoft = nullptr;
FastFillBase *fastFillBase = nullptr;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif





// ------------------------------------------------------------------------------
extern "C" DLLEXPORT void fastFillCloth_new() {
    if (!fastFillCloth)
        fastFillCloth = new FastFillCloth{};
}

extern "C" DLLEXPORT void fastFillCloth_set_data(int* edges_in, int NE_in, float* inv_mass_in, int NV_in, float* pos_in, float alpha_in)
{
    fastFillCloth->set_data_v2(edges_in, NE_in, inv_mass_in, NV_in, pos_in, alpha_in);
}


extern "C" DLLEXPORT void fastFillCloth_init_from_python_cache(
    int *adjacent_edge_in,
    int *num_adjacent_edge_in,
    int *adjacent_edge_abc_in,
    int num_nonz_in,
    float *spmat_data_in,
    int *spmat_indices_in,
    int *spmat_indptr_in,
    int *spmat_ii_in,
    int *spmat_jj_in,
    int NE_in,
    int NV_in)
{
    fastFillCloth->init_from_python_cache_v2(adjacent_edge_in,
                                     num_adjacent_edge_in,
                                     adjacent_edge_abc_in,
                                     num_nonz_in,
                                     spmat_data_in,
                                     spmat_indices_in,
                                     spmat_indptr_in,
                                     spmat_ii_in,
                                     spmat_jj_in,
                                     NE_in,
                                     NV_in);
}

extern "C" DLLEXPORT void fastFillCloth_run(float* pos_in) {
    fastFillCloth->run(pos_in);
}

extern "C" DLLEXPORT void fastFillCloth_fetch_A_data(float* data) {
    fastFillCloth->fetch_A_data(data);
}




// ------------------------------------------------------------------------------
extern "C" DLLEXPORT void fastFillSoft_new() {
    if (!fastFillSoft)
        fastFillSoft = new FastFillSoft{};
}

extern "C" DLLEXPORT void fastFillSoft_set_data(int* tet_in, int NT_in, float* inv_mass_in, int NV_in, float* pos_in, float* alpha_tilde_in)
{
    fastFillSoft->set_data_v2(tet_in, NT_in, inv_mass_in, NV_in, pos_in, alpha_tilde_in);
}


extern "C" DLLEXPORT void fastFillSoft_init_from_python_cache_lessmem(
        const int NT_in,
        const int MAX_ADJ_in,
        const float* data_in,
        const int* indices_in,
        const int* indptr_in,
        const int* ii_in,
        const int num_nonz_in
        )
{
    fastFillSoft->init_from_python_cache_lessmem(
        NT_in,
        MAX_ADJ_in,
        data_in,
        indices_in,
        indptr_in,
        ii_in,
        num_nonz_in
        );
}

extern "C" DLLEXPORT void fastFillSoft_run(float* pos_in, float* gradC_in) {
    fastFillSoft->run(pos_in, gradC_in);
}

extern "C" DLLEXPORT void fastFillSoft_fetch_A_data(float* data) {
    fastFillSoft->fetch_A_data(data);
}

} // namespace fastmg