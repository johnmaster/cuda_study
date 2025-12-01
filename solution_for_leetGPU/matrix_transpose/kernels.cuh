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

#define WARP_SIZE 256
#define WAPR_SIZE_S 16
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

struct Kernel {
    const char* name;
    void (*func)(float*, float*, int, int);  
};

void benchMark(int M, int N);
