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

// Solve function: computes sum of 2D subarray input[S_ROW..E_ROW][S_COL..E_COL] (inclusive)
// d_input: input 2D array on device (stored in row-major order, size N x M)
// d_output: pointer to single int32_t on device where result is stored
// N: number of rows in input array
// M: number of columns in input array
// S_ROW: start row index (inclusive, 0-based)
// E_ROW: end row index (inclusive, 0-based)
// S_COL: start column index (inclusive, 0-based)
// E_COL: end column index (inclusive, 0-based)
void solve(int32_t* d_input, int32_t* d_output, size_t N, size_t M,
           size_t S_ROW, size_t E_ROW, size_t S_COL, size_t E_COL);
