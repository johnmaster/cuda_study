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
            fprintf(stderr, "CUDA error %d: %s at %s:%d\n", err,               \
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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

struct Kernel {
    const char* name;
    void (*func)(float*, float*, float*, int, int);  // A, x, y, M, K
};

void benchMark(int M, int K);

