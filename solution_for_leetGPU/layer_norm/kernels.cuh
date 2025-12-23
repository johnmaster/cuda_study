#ifndef LAYERNORM_CUH
#define LAYERNORM_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cublas_v2.h>
#include <algorithm>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// LayerNorm: output[i] = gamma * (input[i] - mean) / sqrt(variance + epsilon) + beta
// Input: [batch_size, hidden_dim]
// Output: [batch_size, hidden_dim]

// Naive version: each thread handles one element
void layerNormNaive(const float* input, float* output, 
                    const float* gamma, const float* beta,
                    int batch_size, int hidden_dim, float epsilon);

// Block-level reduction: each block handles one row
void layerNormBlockReduction(const float* input, float* output,
                              const float* gamma, const float* beta,
                              int batch_size, int hidden_dim, float epsilon);

// Warp-level reduction using warp shuffle
void layerNormWarpReduction(const float* input, float* output,
                            const float* gamma, const float* beta,
                            int batch_size, int hidden_dim, float epsilon);

// Vectorized load/store version
void layerNormVectorized(const float* input, float* output,
                         const float* gamma, const float* beta,
                         int batch_size, int hidden_dim, float epsilon);

// Welford's online algorithm for numerical stability
void layerNormWelford(const float* input, float* output,
                      const float* gamma, const float* beta,
                      int batch_size, int hidden_dim, float epsilon);

// CPU reference implementation
void layerNormCPU(const float* input, float* output,
                  const float* gamma, const float* beta,
                  int batch_size, int hidden_dim, float epsilon);

// Benchmark function
void benchMark(int batch_size, int hidden_dim);

#endif // LAYERNORM_CUH

