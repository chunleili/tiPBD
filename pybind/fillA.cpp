# include <vector>
# include <array>
# include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "Eigen/Eigen"

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat; // 假设您的矩阵是 float 类型
typedef SpMat::StorageIndex StorageIndexType; // 使用 Eigen 的存储索引类型#include <Eigen/Sparse>
using Vec3f = Eigen::Vector3f;
namespace py = pybind11;

float length(const Vec3f &vec)
{
    return vec.norm();
}

Vec3f normalize(const Vec3f &vec)
{
    return vec.normalized();
}

// from scipy to eigen
SpMat fill_csr(int nrows, int ncols, int nnz, 
                std::vector<int> & indptr, std::vector<int> & indices, std::vector<float> & data)
{
    SpMat m;
    m.resize(nrows, ncols);
    m.makeCompressed();
    m.resizeNonZeros(nnz);

    memcpy((void*)(m.valuePtr()), (void*)(data.data()), sizeof(float) * nnz);
    memcpy((void*)(m.outerIndexPtr()), (void*)(indptr.data()), sizeof(StorageIndexType) * indptr.size());
    memcpy((void*)(m.innerIndexPtr()), (void*)(indices.data()), sizeof(StorageIndexType) * nnz);

    m.finalize();

    // std::cout << "m.rows: " << m.rows() << " m.cols: " << m.cols() << " m.nonZeros: " << m.nonZeros() << std::endl;
    // std::cout<< m << std::endl;
    return m;
}



void pass_vec3f(std::vector<std::array<float, 3>> &v)
{
    for(auto &i:v)
    {
        std::cout << i[0] << " " << i[1] << " " << i[2] << std::endl;
    }
}

void pass_vec2i(std::vector<std::array<int, 2>> &v)
{
    for(auto &i:v)
    {
        std::cout << i[0] << " " << i[1] << std::endl;
    }
}



void compute_C_and_gradC(   std::vector<std::array<int, 2>> &edge, 
                            std::vector<Vec3f> &pos, 
                            std::vector<float> &constraints, 
                            std::vector<float> &rest_len,
                            std::vector<std::array<Vec3f, 2>>  &gradC
                            )
{
    for (int i = 0; i < edge.size(); i++)
    {
        int idx0 = edge[i][0];
        int idx1 = edge[i][1];
        Vec3f dis = pos[idx0] - pos[idx1];
        constraints[i] = length(dis) - rest_len[i];
        Vec3f g = normalize(dis);

        gradC[i][0] = g;
        gradC[i][1] = -g;

        std::cout<<"gradC["<<i<<"][0]: "<<gradC[i][0][0]<<" "<<gradC[i][0][1]<<" "<<gradC[i][0][2]<<std::endl;  
    }
}


void spgemm(int nrows, int ncols, int nnz, 
                std::vector<int> & indptr, std::vector<int> & indices, std::vector<float> & data,
                int nrow2, int ncol2, int nnz2,
                std::vector<int> & indptr2, std::vector<int> & indices2, std::vector<float> & data2,
                std::vector<int> & indptr3, std::vector<int> & indices3, std::vector<float> & data3)
{
    SpMat m1 = fill_csr(nrows, ncols, nnz, indptr, indices, data);
    SpMat m2 = fill_csr(nrow2, ncol2, nnz2, indptr2, indices2, data2);
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
    SpMat m3 = m1 * m2;
    std::cout << m3 << std::endl;
}


void spGMGT_plus_alpha(int nrows, int ncols, int nnz, 
                std::vector<int> & indptr, std::vector<int> & indices, std::vector<float> & data,
                int nrow2, int ncol2, int nnz2,
                std::vector<int> & indptr2, std::vector<int> & indices2, std::vector<float> & data2,
                std::vector<int> & indptr3, std::vector<int> & indices3, std::vector<float> & data3,
                float alpha)
{
    SpMat m1 = fill_csr(nrows, ncols, nnz, indptr, indices, data);
    SpMat m2 = fill_csr(nrow2, ncol2, nnz2, indptr2, indices2, data2);
    SpMat m3 = m1 * m2 * m1.transpose();
    m3.diagonal().array() += alpha;
    std::cout << m3 << std::endl;

    // https://stackoverflow.com/a/51939595/19253199
    // Eigen::Map<Eigen::SparseMatrix<float>> sparse_map(num_rows, num_cols, num_non_zeros,
                            //  original_outer_index_ptr, original_inner_index_ptr,
                            //  original_values_ptr);
}


PYBIND11_MODULE(fillA, m) {
    m.def("fill_csr", &fill_csr, " ");
    m.def("pass_vec3f", &pass_vec3f, " ");
    m.def("pass_vec2i", &pass_vec2i, " ");
    m.def("compute_C_and_gradC", &compute_C_and_gradC, " ");
    m.def("spgemm", &spgemm, " ");
    m.def("spGMGT_plus_alpha", &spGMGT_plus_alpha, " ");
}