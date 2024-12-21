#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <utility>
#include "Vec.h"


namespace fastmg {

// Data of csr matrix
template <class T>
struct CSR {
    Vec<int> indices;
    Vec<T> data;
    Vec<int> indptr;
    int64_t nrows;
    int64_t ncols;
    int64_t numnonz;

    CSR() noexcept : nrows(0), ncols(0), numnonz(0) {}

    void assign(T const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
        indices.resize(nind);
        indptr.resize(nptr);
        data.resize(ndat);
        CHECK_CUDA(cudaMemcpy(data.data(), datap, data.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indices.data(), indicesp, indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indptr.data(), indptrp, indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        nrows = rows;
        ncols = cols;
        numnonz = nnz;
    }

    void assign_v2(T const *datap,  int const *indicesp,  int const *indptrp, size_t rows, size_t cols, size_t nnz) {
        int ndat = nnz;
        int nind = nnz;
        int nptr = rows + 1;
        indices.resize(nind);
        indptr.resize(nptr);
        data.resize(ndat);
        CHECK_CUDA(cudaMemcpy(data.data(), datap, data.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indices.data(), indicesp, indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indptr.data(), indptrp, indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        nrows = rows;
        ncols = cols;
        numnonz = nnz;
    }

    void resize(size_t rows, size_t cols, size_t nnz) {
        nrows = rows;
        ncols = cols;
        numnonz = nnz;
        data.resize(nnz);
        indices.resize(nnz);
        indptr.resize(rows + 1);
    }

    void tohost(std::vector<T> &data_host, std::vector<int> &indices_host, std::vector<int> &indptr_host) const {
        data_host.resize(data.size());
        indices_host.resize(indices.size());
        indptr_host.resize(indptr.size());
        CHECK_CUDA(cudaMemcpy(data_host.data(), data.data(), data.size() * sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices_host.data(), indices.data(), indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr_host.data(), indptr.data(), indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    void fetch(std::vector<T> &data_host, std::vector<int> &indices_host, std::vector<int> &indptr_host) const {
        data_host.resize(data.size());
        indices_host.resize(indices.size());
        indptr_host.resize(indptr.size());
        CHECK_CUDA(cudaMemcpy(data_host.data(), data.data(), data.size() * sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indices_host.data(), indices.data(), indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(indptr_host.data(), indptr.data(), indptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }


};


} // namespace fastmg