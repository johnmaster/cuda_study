#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <cstdint>

#define CHECK_CUDA(call)                                                            \
    do {                                                                            \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                      << cudaGetErrorString(err) << std::endl;                      \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

constexpr int BLOCK_SIZE = 256;

template <typename Func>
double benchmark(Func func, int epochs = 100) {
    func();
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epochs; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / epochs;
}

void rand_init(int32_t* h_data, size_t n, int32_t num_bins);
int32_t* alloc_host(size_t n);
int32_t* alloc_device(size_t n);
void free_host(int32_t* p);
void free_device(int32_t* p);

// Solve function: computes histogram of input array
// input: input array on device (size N)
// histogram: output histogram array on device (size num_bins), must be pre-allocated
// N: number of elements in input array
// num_bins: number of histogram bins (counts values in range [0, num_bins))
void solve(const int32_t* input, int32_t* histogram, size_t N, int32_t num_bins);
