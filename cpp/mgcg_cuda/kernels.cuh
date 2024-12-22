#pragma once
#include <cuda_runtime.h>

extern __global__ void fill_A_CSR_cloth_kernel(
    float *data,
    const int *indptr,
    const int *indices,
    const int *ii,
    const int *jj,
    const int *adjacent_edge_abc,
    const int *num_adjacent_edge,
    const int num_nonz,
    const float *inv_mass,
    const float alpha,
    const int NV,
    const int NE,
    const int *edge,
    const float *pos);

extern __global__ void fill_A_CSR_soft_lessmem_kernel(
    float *data,
    const int *indptr,
    const int *indices, //jj is the same as indices
    const int *ii,
    const int nnz,
    const float *inv_mass,
    const float *alpha_tilde,
    const int NV,
    const int NT,
    const int MAX_ADJ,
    const int *tet,
    const float *pos,
    const float *gradC
    );
// weighted Jacobi for csr matrix
// https://en.wikipedia.org/wiki/Jacobi_method#Weighted_Jacobi_method
// https://stackoverflow.com/questions/78057439/jacobi-algorithm-using-cuda
// https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/amg_core/relaxation.h#L232
// i: row index, j: col index, n: data/indices index
// rsum: sum of off-diagonal elements
extern __global__ void weighted_jacobi_kernel(float *x, float *x_old, const float *b, float *data, int *indices, int *indptr, int nrows, float omega);
extern __global__ void copy_field(float *dst, const float *src, int size);
extern __global__ void calc_diag_inv_kernel(float *diag_inv, const float *data, const int *indices, const int *indptr, const int nrows);
// diag(A)^-1 * A, which is equal to A each row is scaled by the 1./(diagonal element)
extern __global__ void scale_csr_by_row(float *data_new, float *data, const int *indices, const int *indptr, const int nrows, float* diag_inv);
extern __global__ void get_diag_kernel(float *diag, const float *data, const int *indices, const int *indptr, const int nrows);
extern __global__ void fill_sequence_kernel(int *vec, int n);
extern __global__ void get_Aoff_kernel(float *data, const int *indices, const int *indptr, const int nrows);
// parallel gauss seidel
// https://erkaman.github.io/posts/gauss_seidel_graph_coloring.html
// https://gist.github.com/Erkaman/b34b3531e209a1db38e259ea53ff0be9#file-gauss_seidel_graph_coloring-cpp-L101  
extern __global__ void multi_color_gauss_seidel_kernel(float *x, const float *b, float *data, int *indices, int *indptr, int nrows, int *colors, int color);

    
void fill_A_CSR_soft_lessmem_cuda(
    float *data, int *indptr, int *indices,int *d_ii,
    int num_nonz, float *d_inv_mass, float *d_alpha_tilde,
    int NV, int NT, int MAX_ADJ,
    int *d_tet, float *d_pos, float *d_gradC);

void fill_A_CSR_cloth_cuda(
    float *data, int *indptr, int *indices,
    const int *d_ii,
    const int *d_jj,
    const int *d_adjacent_edge_abc,
    const int *d_num_adjacent_edge,
    const int num_nonz,
    const float *d_inv_mass,
    const float alpha_tilde,
    const int NV,
    const int NE,
    const int *d_edges,
    const float *d_pos);


// get diagonal inverse of A, fill into a vector
void calc_diag_inv_cuda(float *diag_inv, const float *data, const int *indices, const int *indptr, const int nrows);

// get Aoff by set diagonal of A to 0
void get_Aoff_cuda(float *Aoff_data, const int *indices, const int *indptr, const int nrows, const int numnonz);
