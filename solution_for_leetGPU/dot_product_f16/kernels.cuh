#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>

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

// Random initialization for half precision arrays (host side uses float conversion)
void rand_init(half* h_data, size_t n);

// Memory allocation/deallocation utilities for half precision
half* alloc_host(size_t n);
half* alloc_device(size_t n);
void free_host(half* p);
void free_device(half* p);

// Solve function: computes dot product of two FP16 vectors
// d_a, d_b: input vectors on device (half precision)
// d_output: pointer to single half on device where result is stored
// n: number of elements in each vector
// 
// Accumulation is done in FP32 for precision, final result stored as half
void solve(const half* d_a, const half* d_b, half* d_output, size_t n);
