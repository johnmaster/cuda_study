#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdint>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << cudaGetErrorString(err)                                  \
                      << " (code " << static_cast<int>(err) << ")" << std::endl;  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int RADIX_BITS = 4;        // 每次处理 4 位
constexpr int RADIX = 1 << RADIX_BITS;  // 16 个桶

template <typename Func>
double benchmark(Func&& func, int epoches = 100) {
    func();
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epoches; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / epoches;
}

void rand_init(uint32_t* h_data, size_t n);
uint32_t* alloc_host(size_t n);
uint32_t* alloc_device(size_t n);
void free_host(uint32_t* p);
void free_device(uint32_t* p);

void radix_sort_gpu(uint32_t* d_input, uint32_t* d_output, size_t n);
