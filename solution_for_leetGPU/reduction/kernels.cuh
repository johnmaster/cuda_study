#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %d: %s at %s:%d\n", err,                \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

#define BLOCK_SIZE 256
#define BLOCK_SIZE_S 32


void rand_init(float* h_data, size_t n);
float* alloc_host(size_t n);
float* alloc_device(size_t n);
void free_host(float* p);
void free_device(float* p);

__global__ void reduce_naive(float* device_out, float* device_in, size_t n);
__global__ void reduce_no_divergence(float* device_out, float* device_in, size_t n);
__global__ void reduce_sequential(float* device_out, float* device_in, size_t n);
__global__ void reduce_first_add(float* device_out, float* device_in, size_t n);
__global__ void reduce_unroll_last_warp(float* device_out, float* device_in, size_t n);
__global__ void reduce_warp_shuffle(float* device_out, float*device_in, size_t n);
__global__ void reduce_vectorized(float* device_out, float* device_in, size_t n);
__global__ void reduce_float4(float* device_out, float* device_in, size_t n);
__global__ void reduce_float4x2(float* device_out, float* device_in, size_t n);

template<typename Func>
double benchmark(Func func, int runs = 100)
{
    cudaEvent_t start, stop;
    CHECK_CUDA( cudaEventCreate(&start) );
    CHECK_CUDA( cudaEventCreate(&stop) );

    // warm up
    for (int i = 0; i < 10; ++i) func();

    CHECK_CUDA( cudaEventRecord(start) );
    for (int i = 0; i < runs; ++i) func();
    CHECK_CUDA( cudaEventRecord(stop) );
    CHECK_CUDA( cudaEventSynchronize(stop) );

    float ms = 0;
    CHECK_CUDA( cudaEventElapsedTime(&ms, start, stop) );
    CHECK_CUDA( cudaEventDestroy(start) );
    CHECK_CUDA( cudaEventDestroy(stop) );

    return ms * 1000.0 / runs;   // 返回平均微秒
}
