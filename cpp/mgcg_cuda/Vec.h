#pragma once

#include <cuda_runtime.h>
#include <utility>
#include "cuda_utils.cuh"

namespace fastmg {

template <class T>
struct Vec {
    T *m_data;
    size_t m_size;
    size_t m_cap;

    Vec() noexcept : m_data(nullptr), m_size(0), m_cap(0) {
    }

    Vec(Vec &&that) noexcept : m_data(that.m_data), m_size(that.m_size), m_cap(that.m_cap) {
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_cap = 0;
    }

    Vec &operator=(Vec &&that) noexcept {
        if (this == &that) return *this;
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
        m_data = that.m_data;
        m_size = that.m_size;
        m_cap = that.m_cap;
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_cap = 0;
        return *this;
    }

    void swap(Vec &that) noexcept {
        std::swap(m_data, that.m_data);
        std::swap(m_size, that.m_size);
        std::swap(m_cap, that.m_cap);
    }

    ~Vec() noexcept {
        if (m_data)
            CHECK_CUDA(cudaFree(m_data));
        m_data = nullptr;
    }

    void resize(size_t new_size) {
        bool change = m_cap < new_size;
        if (change) {
            if (m_data)
                CHECK_CUDA(cudaFree(m_data));
            m_data = nullptr;
            CHECK_CUDA(cudaMalloc(&m_data, sizeof(T) * new_size));
            m_cap = new_size;
        }
        if (m_size != new_size || change) {
            m_size = new_size;
        }
    }

    // host to device
    void assign(T const *data, size_t size) {
        resize(size);
        CHECK_CUDA(cudaMemcpy(m_data, data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    // device to host
    void tohost(std::vector<T> &data_host) const{
        data_host.resize(size());
        CHECK_CUDA(cudaMemcpy(data_host.data(), m_data, sizeof(T) * size(), cudaMemcpyDeviceToHost));
    }

    size_t size() const noexcept {
        return m_size;
    }

    T const *data() const noexcept {
        return m_data;
    }

    T *data() noexcept {
        return m_data;
    }
};

} // namespace