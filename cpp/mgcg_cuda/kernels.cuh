#include <cuda_runtime.h>

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



__global__ void fill_A_CSR_kernel(
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
