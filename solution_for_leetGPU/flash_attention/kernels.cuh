/**
 * Flash Attention Implementation
 * 
 * 核心思想：
 * 1. 不存储 N×N 的注意力矩阵 S (节省 O(N²) 内存)
 * 2. 将 Q, K, V 分块 (tiling) 处理
 * 3. 使用 Online Softmax 动态更新 max 和 sum
 * 4. IO 感知：减少 HBM 访问，尽量在 SRAM 中计算
 * 
 * 复杂度对比：
 * - 标准 Attention: O(N²) 内存，3 passes over N² data
 * - Flash Attention: O(N) 内存，1 pass over N² data (分块)
 */

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

// ============================================================================
// 配置参数
// ============================================================================

// Block tile sizes
constexpr int Br = 32;   // Q 的行块大小 (每个 block 处理 Br 行 query)
constexpr int Bc = 32;   // K/V 的列块大小 (每次加载 Bc 个 key/value)
constexpr int Bd = 64;   // head dimension (支持的最大 d_h)

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

// Memory allocation helpers
void rand_init(float* h_data, size_t n);
float* alloc_host(size_t n);
float* alloc_device(size_t n);
void free_host(float* p);
void free_device(float* p);

/**
 * Flash Attention Forward Pass
 * 
 * 算法流程 (针对一行 query q_i):
 * 1. 初始化: m_i = -inf, l_i = 0, O_i = 0
 * 2. 遍历 K, V 的每个块 j:
 *    a. 计算 S_ij = q_i @ K_j^T / sqrt(d)
 *    b. 计算局部 max: m_ij = max(S_ij)
 *    c. 更新全局 max: m_new = max(m_i, m_ij)
 *    d. 计算 P_ij = exp(S_ij - m_new)
 *    e. 计算局部 sum: l_ij = sum(P_ij)
 *    f. 更新输出: O_i = O_i * exp(m_i - m_new) + P_ij @ V_j
 *    g. 更新累加器: l_i = l_i * exp(m_i - m_new) + l_ij
 *    h. 更新 max: m_i = m_new
 * 3. 最终归一化: O_i = O_i / l_i
 * 
 * @param Q: Query [B, N, d]
 * @param K: Key [B, N, d]  
 * @param V: Value [B, N, d]
 * @param O: Output [B, N, d]
 * @param B: Batch size
 * @param N: Sequence length
 * @param d: Head dimension
 */
void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d
);

/**
 * Multi-Head Flash Attention
 * 
 * @param Q: Query [B, N, d_model]
 * @param K: Key [B, N, d_model]
 * @param V: Value [B, N, d_model]
 * @param O: Output [B, N, d_model]
 * @param B: Batch size
 * @param N: Sequence length
 * @param d_model: Model dimension
 * @param num_heads: Number of attention heads
 */
void flash_attention_multihead(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d_model, int num_heads
);

// Standard attention for comparison
void standard_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d
);

