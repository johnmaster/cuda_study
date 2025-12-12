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

void rand_init(int32_t* h_data, size_t n);
int32_t* alloc_host(size_t n);
int32_t* alloc_device(size_t n);
void free_host(int32_t* p);
void free_device(int32_t* p);

// Solve function: computes sum of subarray input[S..E] (inclusive)
// d_input: input array on device
// d_output: pointer to single int32_t on device where result is stored
// n: number of elements in input array
// S: start index (inclusive, 0-based)
// E: end index (inclusive, 0-based)
void solve(int32_t* d_input, int32_t* d_output, size_t n, size_t S, size_t E);
