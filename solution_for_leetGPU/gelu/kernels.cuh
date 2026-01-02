#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>

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

// Constants for GELU approximation
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/π)
constexpr float GELU_COEFF = 0.044715f;

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

void rand_init(float* h_data, size_t n);
float* alloc_host(size_t n);
float* alloc_device(size_t n);
void free_host(float* p);
void free_device(float* p);

// Solve function: computes GELU activation of input array
// input: input array on device (size N)
// output: output GELU array on device (size N), must be pre-allocated
// N: number of elements in input array
void solve(const float* input, float* output, size_t N);

