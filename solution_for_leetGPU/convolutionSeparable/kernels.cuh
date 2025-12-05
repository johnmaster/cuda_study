#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %d: %s at %s:%d\n", err,                \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

constexpr int KERNEL_RADIUS = 8;
constexpr int KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1;

#define ROW_BLOCKDIM_X      16
#define ROW_BLOCKDIM_Y      4
#define ROW_RESULT_STEPS    8
#define ROWS_HALO_STEPS     1

#define COLUMNS_BLOCKDIM_X      16
#define COLUMNS_BLOCKDIM_Y      8
#define COLUMNS_RESULT_STEPS    8
#define COLUMNS_HALO_STEPS      1

extern "C" void convolutionRowCPU(float* host_dst, float* host_src, float* host_kernel,
                                  int imageW, int imageH, int kernelR);
extern "C" void convolutionColumnCPU(float* host_dst, float* host_src, float* host_kernel,
                                     int imageW, int imageH, int kernelR);

extern "C" void setConvolutionKernel(float* host_kernel);
extern "C" void convolutionRowsGPU(float* device_dst, float* device_src, int imageW, int imageH);
extern "C" void convolutionColumnsGPU(float* device_dst, float* device_src, int imageW, int imageH);
