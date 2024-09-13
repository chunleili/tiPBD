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