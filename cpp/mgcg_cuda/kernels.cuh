#include <cuda_runtime.h>

#define USE_LESSMEM 1

// normalize(v1-v2)
__device__ float3 inline d_normalize_diff(float3 v1, float3 v2)
{
    float3 diff = make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    float rnorm = rsqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
    return make_float3(diff.x * rnorm, diff.y * rnorm, diff.z * rnorm);
}

// dot product
__device__ float inline d_dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// norm sqr 
__device__ float inline d_norm_sqr(float3 a)
{
    return a.x*a.x + a.y*a.y + a.z*a.z;
}


__global__ void fill_A_CSR_cloth_kernel(
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
    const float *pos)
{
    size_t cnt = blockIdx.x * blockDim.x + threadIdx.x;
    if(cnt >= num_nonz)
    {
        return;
    }
    else
    {
        int row_idx = ii[cnt]; // row index
        int j = jj[cnt]; // col index
        if (row_idx==j) // diag
        {
            data[cnt] = inv_mass[edge[row_idx*2 + 0]] + inv_mass[edge[row_idx*2 + 1]] + alpha;
        }
        else
        {
            int k = cnt - indptr[row_idx]; // k-th element in row row_idx
            int n = num_adjacent_edge[row_idx];
            int width = 3*20;
            if (k >= n)
            {
                return;
            }
            int a = adjacent_edge_abc[row_idx* width + k*3 + 0];
            int b = adjacent_edge_abc[row_idx* width + k*3 + 1];
            int c = adjacent_edge_abc[row_idx* width + k*3 + 2];

            float3 pa = make_float3(pos[a*3 + 0], pos[a*3 + 1], pos[a*3 + 2]);
            float3 pb = make_float3(pos[b*3 + 0], pos[b*3 + 1], pos[b*3 + 2]);
            float3 pc = make_float3(pos[c*3 + 0], pos[c*3 + 1], pos[c*3 + 2]);
            float3 g_ab = d_normalize_diff(pa,pb);
            float3 g_ac = d_normalize_diff(pa,pc);
            float dot = d_dot(g_ab, g_ac);
            float offdiag = inv_mass[a] * dot;
            data[cnt] = offdiag; 
        }
    }
}



#ifndef USE_LESSMEM
__global__ void fill_A_CSR_soft_kernel(
    float *data,
    const int *indptr,
    const int *indices,
    const int *ii,
    const int *jj,
    const int *adjacent,
    const int *num_adjacent,
    const int nnz,
    const float *inv_mass,
    const float *alpha_tilde,
    const int NV,
    const int NT,
    const int MAX_ADJ,
    const int *tet,
    const float *pos,
    const float *gradC,
    const int * n_shared_v,
    const int * shared_v,
    const int8_t * shared_v_order_in_cur,
    const int8_t * shared_v_order_in_adj
    )
{
    size_t cnt = blockIdx.x * blockDim.x + threadIdx.x;


    if(cnt >= nnz)
    {
        return;
    }

    int i = ii[cnt]; // row index
    int j = jj[cnt]; // col index


    if(i==j) // diag
    {
        float m1 = inv_mass[tet[i*4 + 0]];
        float m2 = inv_mass[tet[i*4 + 1]];
        float m3 = inv_mass[tet[i*4 + 2]];
        float m4 = inv_mass[tet[i*4 + 3]];
        float alpha = alpha_tilde[i];
        float3 g1 = make_float3(gradC[i*4*3 + 0*3 + 0], gradC[i*4*3 + 0*3 + 1], gradC[i*4*3 + 0*3 + 2]);
        float3 g2 = make_float3(gradC[i*4*3 + 1*3 + 0], gradC[i*4*3 + 1*3 + 1], gradC[i*4*3 + 1*3 + 2]);
        float3 g3 = make_float3(gradC[i*4*3 + 2*3 + 0], gradC[i*4*3 + 2*3 + 1], gradC[i*4*3 + 2*3 + 2]);
        float3 g4 = make_float3(gradC[i*4*3 + 3*3 + 0], gradC[i*4*3 + 3*3 + 1], gradC[i*4*3 + 3*3 + 2]);
        float diag = m1* d_norm_sqr(g1) + m2* d_norm_sqr(g2) + m3* d_norm_sqr(g3) + m4* d_norm_sqr(g4) + alpha;
        data[cnt] = diag;
    }
    else // offdiag 
    {
        int k = cnt - indptr[i]; // k-th element in row i

        float offdiag = 0.0;
        for(int kv=0; kv < n_shared_v[i* MAX_ADJ + k]; kv++)
        {
            int o1 = shared_v_order_in_cur[i* MAX_ADJ* 3 + k * 3 + kv]; // shared vertex order in current tet
            int o2 = shared_v_order_in_adj[i* MAX_ADJ* 3 + k * 3 + kv]; // shared vertex order in adjacent tet
            int sv = shared_v[i* MAX_ADJ* 3 + k * 3 + kv]; // shared vertex index
            float sm = inv_mass[sv]; //shared vertex inv mass
            float3 go1 = make_float3(gradC[i*4*3 + o1*3 + 0], gradC[i*4*3 + o1*3 + 1], gradC[i*4*3 + o1*3 + 2]);
            float3 go2 = make_float3(gradC[j*4*3 + o2*3 + 0], gradC[j*4*3 + o2*3 + 1], gradC[j*4*3 + o2*3 + 2]);
            offdiag += sm * d_dot(go1, go2);
        }
        data[cnt] = offdiag;
    }
}
#endif



// intersect
/// 求两个长度为4的数组的交集
/// @param a: 4个顶点的id
/// @param b: 4个顶点的id
/// @return n: 有几个共享的顶点， 0, 1, 2, 3
/// @return shared_v: 共享的顶点id, 最多有三个共享的顶点，分别是shared_v[0], shared_v[1], shared_v[2]
/// @return o1: 共享的顶点是当前ele的第几个顶点，分别对应三个共享的顶点
/// @return o2: 共享的顶点是邻接ele的第几个顶点，分别对应三个共享的顶点
__device__ int inline d_compare_find_shared_4x4(int* a, int* b, int* shared_v, int* o1, int* o2)
{
    int n=0; 
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            if(a[i] == b[j])
            {
                shared_v[n] = a[i];
                o1[n] = i;
                o2[n] = j;
                n++;
            }
        }
    }
    return n;
}


__global__ void fill_A_CSR_soft_lessmem_kernel(
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
    )
    // const int *jj,
    // const int *adjacent,
    // const int *num_adjacent,
    // const int * n_shared_v,
    // const int * shared_v,
    // const int8_t * shared_v_order_in_cur,
    // const int8_t * shared_v_order_in_adj
    // )
{
    size_t cnt = blockIdx.x * blockDim.x + threadIdx.x;


    if(cnt >= nnz)
    {
        return;
    }

    int i = ii[cnt]; // row index
    int j = indices[cnt]; // col index


    if(i==j) // diag
    {
        float m1 = inv_mass[tet[i*4 + 0]];
        float m2 = inv_mass[tet[i*4 + 1]];
        float m3 = inv_mass[tet[i*4 + 2]];
        float m4 = inv_mass[tet[i*4 + 3]];
        float alpha = alpha_tilde[i];
        float3 g1 = make_float3(gradC[i*4*3 + 0*3 + 0], gradC[i*4*3 + 0*3 + 1], gradC[i*4*3 + 0*3 + 2]);
        float3 g2 = make_float3(gradC[i*4*3 + 1*3 + 0], gradC[i*4*3 + 1*3 + 1], gradC[i*4*3 + 1*3 + 2]);
        float3 g3 = make_float3(gradC[i*4*3 + 2*3 + 0], gradC[i*4*3 + 2*3 + 1], gradC[i*4*3 + 2*3 + 2]);
        float3 g4 = make_float3(gradC[i*4*3 + 3*3 + 0], gradC[i*4*3 + 3*3 + 1], gradC[i*4*3 + 3*3 + 2]);
        float diag = m1* d_norm_sqr(g1) + m2* d_norm_sqr(g2) + m3* d_norm_sqr(g3) + m4* d_norm_sqr(g4) + alpha;
        data[cnt] = diag;
    }
    else // offdiag 
    {
        int k = cnt - indptr[i]; // k-th element in row i

        float offdiag = 0.0;

        int a[4] = {tet[i*4 + 0], tet[i*4 + 1], tet[i*4 + 2], tet[i*4 + 3]};
        int b[4] = {tet[j*4 + 0], tet[j*4 + 1], tet[j*4 + 2], tet[j*4 + 3]};
        int shared_v[3] = {-1,-1,-1};
        int shared_v_order_in_cur[3] = {-1,-1,-1};
        int shared_v_order_in_adj[3] = {-1,-1,-1};

        int n_shared_v = d_compare_find_shared_4x4(a, b, shared_v, shared_v_order_in_cur, shared_v_order_in_adj);

        // for(int kv=0; kv < n_shared_v[i* MAX_ADJ + k]; kv++)
        for(int kv=0; kv < n_shared_v; kv++) //kv: 第几个共享的顶点 0, 1, 2 最多有三个共享的顶点
        {
            int o1 = shared_v_order_in_cur[kv]; // shared vertex order in current tet
            int o2 = shared_v_order_in_adj[kv]; // shared vertex order in adjacent tet
            int sv = shared_v[kv]; // shared vertex index

            // int o1 = shared_v_order_in_cur[i* MAX_ADJ* 3 + k * 3 + kv]; // shared vertex order in current tet
            // int o2 = shared_v_order_in_adj[i* MAX_ADJ* 3 + k * 3 + kv]; // shared vertex order in adjacent tet
            // int sv = shared_v[i* MAX_ADJ* 3 + k * 3 + kv]; // shared vertex index
            float sm = inv_mass[sv]; //shared vertex inv mass
            float3 go1 = make_float3(gradC[i*4*3 + o1*3 + 0], gradC[i*4*3 + o1*3 + 1], gradC[i*4*3 + o1*3 + 2]);
            float3 go2 = make_float3(gradC[j*4*3 + o2*3 + 0], gradC[j*4*3 + o2*3 + 1], gradC[j*4*3 + o2*3 + 2]);
            offdiag += sm * d_dot(go1, go2);
        }
        data[cnt] = offdiag;
    }
}






// weighted Jacobi for csr matrix
// https://en.wikipedia.org/wiki/Jacobi_method#Weighted_Jacobi_method
// https://stackoverflow.com/questions/78057439/jacobi-algorithm-using-cuda
// https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/amg_core/relaxation.h#L232
// i: row index, j: col index, n: data/indices index
// rsum: sum of off-diagonal elements
__global__ void weighted_jacobi_kernel(float *x, float *x_old, const float *b, float *data, int *indices, int *indptr, int nrows, float omega) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        float rsum = 0.0;
        float diag = 0.0;
        for (size_t n = indptr[i]; n < indptr[i + 1]; ++n) {
            size_t j = indices[n];
            if (j != i) {
                rsum += data[n] * x_old[j];
            }
            else {
                diag = data[n];
            }
        }
        // FIXME: should use x_new to avoid race condition
        if (diag != 0.0)
        {
            x[i] =  omega / diag * (b[i] - rsum)  + (1.0 - omega) * x_old[i];
        }
    }
}

__global__ void copy_field(float *dst, const float *src, int size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] = src[i];
    }
}


void jacobi_serial(const int Ap[], const int Ap_size,
            const int Aj[], const int Aj_size,
            const float Ax[], const int Ax_size,
                  float  x[], const int  x_size,
            const float  b[], const int  b_size,
                  float temp[], const int temp_size,
            const int row_start,
            const int row_stop,
            const int row_step,
            const float omega)
{
    float one = 1.0;

    for(int i = row_start; i != row_stop; i += row_step) {
        temp[i] = x[i];
    }

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
                rsum += Ax[jj]*temp[j];
        }

        if (diag != (float) 0.0){
            x[i] = (one - omega) * temp[i] + omega * ((b[i] - rsum)/diag);
        }
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



__global__ void calc_diag_inv_kernel(float *diag_inv, const float *data, const int *indices, const int *indptr, const int nrows) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        for (size_t n = indptr[i]; n < indptr[i + 1]; ++n) {
            size_t j = indices[n];
            if (j == i) {
                diag_inv[i] = 1.0/data[n];
            }
        }
    }
}


// diag(A)^-1 * A, which is equal to A each row is scaled by the 1./(diagonal element)
__global__ void scale_csr_by_row(float *data_new, float *data, const int *indices, const int *indptr, const int nrows, float* diag_inv) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        for (size_t n = indptr[i]; n < indptr[i + 1]; ++n) {
            size_t j = indices[n];
             data_new[n] = data[n] * diag_inv[i];
        }
    }
}


__global__ void get_diag_kernel(float *diag, const float *data, const int *indices, const int *indptr, const int nrows) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        for (size_t n = indptr[i]; n < indptr[i + 1]; ++n) {
            size_t j = indices[n];
            if (j == i) {
                diag[i] = data[n];
            }
        }
    }
}


__global__ void fill_sequence_kernel(int *vec, int n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vec[i] = i;
    }
}


__global__ void get_Aoff_kernel(float *data, const int *indices, const int *indptr, const int nrows) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        for (size_t n = indptr[i]; n < indptr[i + 1]; ++n) {
            size_t j = indices[n];
            if (j == i) {
                data[n] = 0.0;
            }
        }
    }
}