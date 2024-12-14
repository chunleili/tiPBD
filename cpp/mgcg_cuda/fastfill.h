#pragma once

#include "fastmg.h"

namespace fastmg {
// Base case for FastFillCloth and FastFillSoft
struct FastFillBase : CusparseWrappers
{
    int num_nonz;
    int nrows, ncols;
    CSR<float> A;
};

struct FastFillCloth : FastFillBase
{
    float alpha;
    int NE;
    int NV;
    Vec<float> d_inv_mass;
    Vec<int> d_ii, d_jj;
    Vec<int> d_edges;
    Vec<float> d_pos;
    Vec<int> d_adjacent_edge_abc;
    Vec<int> d_num_adjacent_edge;

    void fetch_A_data(float *data_in);
    void set_data_v2(int *edges_in, int NE_in, float *inv_mass_in, int NV_in, float *pos_in, float alpha_in);
    void update_pos_py2cu(float *pos_in);
    void init_from_python_cache_v2(
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
        int NV_in);
    void run(float *pos_in);
    void fill_A_CSR_gpu();
}; // FastFillCloth struct


struct FastFillSoft : FastFillBase
{
    int NT;
    int NV;
    int MAX_ADJ;
    Vec<float> d_inv_mass;
    Vec<int> d_ii;
    Vec<float> d_pos;
    Vec<int> d_tet;
    Vec<float> d_gradC;
    Vec<float> d_alpha_tilde;

    void fetch_A_data(float *data_in);
    void set_data_v2(int *tet_in, int NT_in, float *inv_mass_in, int NV_in, float *pos_in, float *alpha_tilde_in);
    void update_pos_and_gradC(float *pos_in, float *gradC_in);
    void init_from_python_cache_lessmem(
        const int NT_in,
        const int MAX_ADJ_in,
        const float *data_in,
        const int *indices_in,
        const int *indptr_in,
        const int *ii_in,
        const int num_nonz_in);
    void run(float *pos_in, float *gradC_in);
    void fill_A_CSR_gpu();
}; // FastFillSoft struct



extern FastFillCloth *fastFillCloth;
extern FastFillSoft *fastFillSoft;
extern FastFillBase *fastFillBase;

} // namespace fastmg