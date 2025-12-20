#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
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

#define BLOCK_SIZE 256
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// 辅助函数
void rand_init(float* h_data, size_t rows, size_t cols);
float* alloc_host(size_t n);
float* alloc_device(size_t n);
void free_host(float* p);
void free_device(float* p);

// 行规约 (Row Reduction): 对每一行进行规约，输出大小为 [rows, 1]
__global__ void reduce_rows_naive(float* device_out, const float* device_in, size_t rows, size_t cols);
__global__ void reduce_rows_optimized(float* device_out, const float* device_in, size_t rows, size_t cols);

// 列规约 (Column Reduction): 对每一列进行规约，输出大小为 [1, cols]
__global__ void reduce_cols_naive(float* device_out, const float* device_in, size_t rows, size_t cols);
__global__ void reduce_cols_optimized(float* device_out, const float* device_in, size_t rows, size_t cols);

// 全局规约 (Global Reduction): 对整个矩阵进行规约，输出单个值
__global__ void reduce_global_stage1(float* device_out, const float* device_in, size_t n);
__global__ void reduce_global_stage2(float* device_out, const float* device_in, size_t n);

// Warp级别的shuffle规约
__device__ float warpReduceShuffle(float val);

// CPU验证函数
void verify_row_reduction(const float* input, const float* output, size_t rows, size_t cols);
void verify_col_reduction(const float* input, const float* output, size_t rows, size_t cols);
void verify_global_reduction(const float* input, float output, size_t n);

// Benchmark模板
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

