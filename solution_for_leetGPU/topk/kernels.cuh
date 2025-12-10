#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

constexpr int BLOCK_SIZE = 256;

void rand_init(float* h_data, size_t n);
float* alloc_host(size_t n);
float* alloc_device(size_t n);
void free_host(float* p);
void free_device(float* p);

template<typename Func>
double benchmark(Func func, int epoches = 100) {
    func();
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epoches; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    return total_us / epoches;
}

void solve(float* input, float* output, size_t n, size_t k);
void solve_bitonic_sort(float* input, float* output, size_t n, size_t k);
//void solve_radix_select(float* input, float* output, size_t n, size_t k);
//void solve_heap_select(float* intput, float* output, size_t n, size_t k);