#include <cuda_runtime.h>
// normalize(v1-v2)
__device__ void inline d_normalize_diff(float v1x, float v1y, float v1z, float v2x, float v2y, float v2z, float *v3x, float *v3y, float *v3z)
{
    float x = v1x - v2x;
    float y = v1y - v2y;
    float z = v1z - v2z;
    float rnorm = rsqrt(x*x + y*y + z*z);
    *v3x = x * rnorm;
    *v3y = y * rnorm;
    *v3z = z * rnorm;
} 

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
    if (cnt < num_nonz) {
        int row_idx = ii[cnt]; // row index
        int j = jj[cnt]; // col index
        int k = cnt - indptr[row_idx]; // k-th element in row row_idx
        if (row_idx==j) // diag
        {
            data[cnt] = inv_mass[edge[row_idx*2 + 0]] + inv_mass[edge[row_idx*2 + 1]] + alpha;
            // printf("cnt=%d, row_idx=%d, j=%d, data=%f\n", cnt, row_idx, j, data[cnt]);
        }
        else
        {
            int n = num_adjacent_edge[row_idx];
            int width = 3*20;

            int a = adjacent_edge_abc[row_idx* width + k*3 + 0];
            int b = adjacent_edge_abc[row_idx* width + k*3 + 1];
            int c = adjacent_edge_abc[row_idx* width + k*3 + 2];

            if(cnt == 0)
            {
                printf("row_idx=%d, n=%d\n", row_idx, n);
                // num_adjacent_edge
                printf("num_adjacent_edge[row_idx]=%d\n", num_adjacent_edge[row_idx]);
                // adjacent_edge_abc
                for(int i=0; i<n; i++)
                {
                    printf("adjacent_edge_abc[%d]=%d, %d, %d\n", i, adjacent_edge_abc[row_idx* width + i*3 + 0], adjacent_edge_abc[row_idx* width + i*3 + 1], adjacent_edge_abc[row_idx* width + i*3 + 2]);
                }

                // print a b c
                printf("a=%d, b=%d, c=%d\n", a, b, c);
            }

            float pa_x = pos[a*3 + 0];
            float pa_y = pos[a*3 + 1];
            float pa_z = pos[a*3 + 2];
            float pb_x = pos[b*3 + 0];
            float pb_y = pos[b*3 + 1];
            float pb_z = pos[b*3 + 2];
            float pc_x = pos[c*3 + 0];
            float pc_y = pos[c*3 + 1];
            float pc_z = pos[c*3 + 2];

            if(cnt == 0)
            {
                printf("pa=%f, %f, %f\n", pa_x, pa_y, pa_z);
                printf("pb=%f, %f, %f\n", pb_x, pb_y, pb_z);
                printf("pc=%f, %f, %f\n", pc_x, pc_y, pc_z);
            }
            
            float g_ab_x, g_ab_y, g_ab_z;
            float x = pa_x - pb_x;  // BUG: WHY use x=1.0, y=1.0, z=1.0 will not cause bug. Use x=pa_x - pb_x, y=pa_y - pb_y, z=pa_z - pb_z will cause bug:Failed to launch kernel (error code: an illegal memory access was encountered)!
            // float y = pa_y - pb_y;
            // float z = pa_z - pb_z;
            // float x=1.0, y=1.0, z=1.0;
            if(cnt == 0)
            {
                printf("x=%f\n", x);
            }

            // float rnorm = rsqrtf(x*x + y*y + z*z);
            // g_ab_x = x * rnorm;
            // g_ab_y = y * rnorm;
            // g_ab_z = z * rnorm;

            // // float3 g_ab = d_normalize_diff(pa, pb);
            // // float3 g_ac = d_normalize_diff(pa, pc);
            // // float offdiag = inv_mass[a] * d_dot(g_ab, g_ac);
            
            // data[cnt] = offdiag; 
        }
    }
}