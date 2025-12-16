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
constexpr int TILE_SIZE = 16;

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

// Memory allocation helpers
void rand_init(float* h_data, size_t n);
float* alloc_host(size_t n);
float* alloc_device(size_t n);
void free_host(float* p);
void free_device(float* p);

/**
 * Multi-Head Self-Attention
 * 
 * Given Q, K, V matrices of shape [B, N, d]:
 * - B: batch size
 * - N: sequence length
 * - d: embedding dimension (must be divisible by num_heads)
 * 
 * For each head h (0 <= h < H):
 *   Q_h, K_h, V_h = Q[:,:,h*d_h:(h+1)*d_h], K[:,:,h*d_h:(h+1)*d_h], V[:,:,h*d_h:(h+1)*d_h]
 *   S_h = Q_h @ K_h^T                       [B, N, N]
 *   S_h = S_h / sqrt(d_h)                   [B, N, N]
 *   A_h = softmax(S_h, dim=-1)              [B, N, N]
 *   O_h = A_h @ V_h                         [B, N, d_h]
 * 
 * Output = concat(O_1, ..., O_H)            [B, N, d]
 * 
 * @param Q: Query matrix on device [B * N * d]
 * @param K: Key matrix on device [B * N * d]
 * @param V: Value matrix on device [B * N * d]
 * @param output: Output matrix on device [B * N * d], must be pre-allocated
 * @param B: Batch size
 * @param N: Sequence length
 * @param d: Embedding dimension
 * @param num_heads: Number of attention heads
 */
void solve(const float* Q, const float* K, const float* V, float* output,
           int B, int N, int d, int num_heads);

