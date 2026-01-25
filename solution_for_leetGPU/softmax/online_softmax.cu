/**
 * Online Softmax Implementation
 * 
 * Online Softmax 将传统的 3-pass 算法优化为 2-pass：
 * - Pass 1: 同时计算 max 和 sum（使用动态修正）
 * - Pass 2: 归一化
 * 
 * 核心公式（合并两个 (m, d) 对）：
 *   m_new = max(m1, m2)
 *   d_new = d1 * exp(m1 - m_new) + d2 * exp(m2 - m_new)
 */

#include "kernels.cuh"

// ============================================================================
// Online Softmax 数据结构
// ============================================================================

struct MD {  // (max, denominator) pair
    float m;  // 当前最大值
    float d;  // exp 累加和（相对于当前 max）
};

// ============================================================================
// Device Functions
// ============================================================================

// 合并两个 MD 对
__device__ __forceinline__ MD reduceMD(MD a, MD b) {
    MD result;
    result.m = fmaxf(a.m, b.m);
    // 修正因子：将两个 d 都调整到新的 max 下
    result.d = a.d * expf(a.m - result.m) + b.d * expf(b.m - result.m);
    return result;
}

// Warp-level MD reduction
__device__ __forceinline__ MD warpReduceMD(MD val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        MD other;
        other.m = __shfl_down_sync(0xffffffff, val.m, offset);
        other.d = __shfl_down_sync(0xffffffff, val.d, offset);
        val = reduceMD(val, other);
    }
    return val;
}

// Block-level MD reduction
__device__ MD blockReduceMD(MD val) {
    __shared__ float shared_m[32];
    __shared__ float shared_d[32];
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // First reduce within warp
    val = warpReduceMD(val);
    
    // Write reduced value from each warp to shared memory
    if (lane == 0) {
        shared_m[wid] = val.m;
        shared_d[wid] = val.d;
    }
    __syncthreads();
    
    // Read from shared memory only if thread is in first warp
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        val.m = shared_m[lane];
        val.d = shared_d[lane];
    } else {
        val.m = -INFINITY;
        val.d = 0.0f;
    }
    
    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceMD(val);
    }
    
    return val;
}

// ============================================================================
// Kernel 1: Online Softmax (Single Block Version)
// 适用于较小的数组，一个 block 处理所有数据
// ============================================================================

__global__ void onlineSoftmaxSingleBlockKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t N
) {
    __shared__ float s_max;
    __shared__ float s_sum;
    
    // ========== Pass 1: Online 计算 (max, sum) ==========
    MD local_md;
    local_md.m = -INFINITY;
    local_md.d = 0.0f;
    
    // 每个线程处理多个元素，在线更新 (m, d)
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float x = input[i];
        // Online update: 新元素 x 对应的 MD 是 (x, 1.0)
        // 因为 exp(x - x) = 1
        MD new_md;
        new_md.m = x;
        new_md.d = 1.0f;
        local_md = reduceMD(local_md, new_md);
    }
    
    // Block-level reduction
    MD block_md = blockReduceMD(local_md);
    
    if (threadIdx.x == 0) {
        s_max = block_md.m;
        s_sum = block_md.d;
    }
    __syncthreads();
    
    float global_max = s_max;
    float global_sum = s_sum;
    
    // ========== Pass 2: 归一化 ==========
    float inv_sum = 1.0f / global_sum;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - global_max) * inv_sum;
    }
}

// ============================================================================
// Kernel 2: Online Softmax (Multi-Block Version)
// 适用于大数组，需要多个 block 协作
// ============================================================================

// Pass 1: 每个 block 计算局部的 (max, d) 对
__global__ void onlineSoftmaxPass1Kernel(
    const float* __restrict__ input,
    float* __restrict__ block_max,
    float* __restrict__ block_sum,
    size_t N
) {
    MD local_md;
    local_md.m = -INFINITY;
    local_md.d = 0.0f;
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // 每个线程处理多个元素
    for (size_t i = tid; i < N; i += stride) {
        float x = input[i];
        MD new_md;
        new_md.m = x;
        new_md.d = 1.0f;
        local_md = reduceMD(local_md, new_md);
    }
    
    // Block-level reduction
    MD block_md = blockReduceMD(local_md);
    
    // 写入 block 结果
    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = block_md.m;
        block_sum[blockIdx.x] = block_md.d;
    }
}

// Pass 1.5: 合并所有 block 的 (max, d) 对
__global__ void onlineSoftmaxReduceBlocksKernel(
    float* __restrict__ block_max,
    float* __restrict__ block_sum,
    int numBlocks
) {
    MD local_md;
    local_md.m = -INFINITY;
    local_md.d = 0.0f;
    
    // 每个线程读取多个 block 的结果
    for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
        MD block_md;
        block_md.m = block_max[i];
        block_md.d = block_sum[i];
        local_md = reduceMD(local_md, block_md);
    }
    
    // Block-level reduction
    MD final_md = blockReduceMD(local_md);
    
    // 写入最终结果
    if (threadIdx.x == 0) {
        block_max[0] = final_md.m;
        block_sum[0] = final_md.d;
    }
}

// Pass 2: 归一化
__global__ void onlineSoftmaxPass2Kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float global_max,
    float global_sum,
    size_t N
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float inv_sum = 1.0f / global_sum;
    
    for (size_t i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - global_max) * inv_sum;
    }
}

// ============================================================================
// Host Wrapper Function
// ============================================================================

void solve_online(const float* input, float* output, size_t N) {
    if (N == 0) {
        return;
    }
    
    // 对于小数组，使用 single-block 版本
    if (N <= BLOCK_SIZE * 32) {
        onlineSoftmaxSingleBlockKernel<<<1, BLOCK_SIZE>>>(input, output, N);
        return;
    }
    
    // 对于大数组，使用 multi-block 版本
    int numBlocks = min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    
    // 分配临时存储
    float* d_block_max;
    float* d_block_sum;
    CHECK_CUDA(cudaMalloc(&d_block_max, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_sum, numBlocks * sizeof(float)));
    
    // Pass 1: 每个 block 计算局部 (max, sum)
    onlineSoftmaxPass1Kernel<<<numBlocks, BLOCK_SIZE>>>(
        input, d_block_max, d_block_sum, N);
    
    // Pass 1.5: 合并所有 block 的结果
    onlineSoftmaxReduceBlocksKernel<<<1, BLOCK_SIZE>>>(
        d_block_max, d_block_sum, numBlocks);
    
    // 获取全局 max 和 sum
    float global_max, global_sum;
    CHECK_CUDA(cudaMemcpy(&global_max, d_block_max, sizeof(float), 
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&global_sum, d_block_sum, sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    // Pass 2: 归一化
    onlineSoftmaxPass2Kernel<<<numBlocks, BLOCK_SIZE>>>(
        input, output, global_max, global_sum, N);
    
    // 释放临时存储
    CHECK_CUDA(cudaFree(d_block_max));
    CHECK_CUDA(cudaFree(d_block_sum));
}

// ============================================================================
// Main: 测试和性能对比
// ============================================================================

// CPU reference
void cpu_softmax_ref(const float* input, float* output, size_t N) {
    if (N == 0) return;
    
    float maxVal = input[0];
    for (size_t i = 1; i < N; i++) {
        maxVal = std::max(maxVal, input[i]);
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    
    for (size_t i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

bool compare(const float* a, const float* b, size_t N, float tol = 1e-5f) {
    for (size_t i = 0; i < N; i++) {
        float diff = std::abs(a[i] - b[i]);
        float maxAbs = std::max(std::abs(a[i]), std::abs(b[i]));
        float effectiveTol = std::max(tol, tol * maxAbs);
        if (diff > effectiveTol) {
            return false;
        }
    }
    return true;
}

void benchmark_online_softmax(size_t N) {
    std::cout << "\n=== Online Softmax Benchmark: N = " << N << " ===" << std::endl;
    
    // 分配内存
    float* h_input = alloc_host(N);
    float* h_output_cpu = alloc_host(N);
    float* h_output_online = alloc_host(N);
    float* h_output_traditional = alloc_host(N);
    
    rand_init(h_input, N);
    
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // CPU reference
    cpu_softmax_ref(h_input, h_output_cpu, N);
    
    // Online Softmax benchmark
    double online_time = benchmark([&]() {
        solve_online(d_input, d_output, N);
    }, 100);
    CHECK_CUDA(cudaMemcpy(h_output_online, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Traditional Softmax benchmark (使用原有的 solve 函数)
    double traditional_time = benchmark([&]() {
        solve(d_input, d_output, N);
    }, 100);
    CHECK_CUDA(cudaMemcpy(h_output_traditional, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool online_correct = compare(h_output_online, h_output_cpu, N);
    bool traditional_correct = compare(h_output_traditional, h_output_cpu, N);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Traditional Softmax: " << traditional_time << " us - " 
              << (traditional_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Online Softmax:      " << online_time << " us - " 
              << (online_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Speedup (Online/Traditional): " << traditional_time / online_time << "x" << std::endl;
    
    // Cleanup
    free_host(h_input);
    free_host(h_output_cpu);
    free_host(h_output_online);
    free_host(h_output_traditional);
    free_device(d_input);
    free_device(d_output);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "============================================" << std::endl;
    std::cout << "        Online Softmax Implementation       " << std::endl;
    std::cout << "============================================" << std::endl;
    
    std::cout << "\n--- 算法说明 ---" << std::endl;
    std::cout << "传统 Softmax: 3 passes (findMax, computeExpSum, normalize)" << std::endl;
    std::cout << "Online Softmax: 2 passes (onlineMaxSum, normalize)" << std::endl;
    std::cout << "\n核心公式:" << std::endl;
    std::cout << "  m_new = max(m1, m2)" << std::endl;
    std::cout << "  d_new = d1 * exp(m1 - m_new) + d2 * exp(m2 - m_new)" << std::endl;
    
    // 各种大小的测试
    benchmark_online_softmax(1 << 10);   // 1K
    benchmark_online_softmax(1 << 14);   // 16K
    benchmark_online_softmax(1 << 16);   // 64K
    benchmark_online_softmax(1 << 18);   // 256K
    benchmark_online_softmax(1 << 20);   // 1M
    benchmark_online_softmax(1 << 22);   // 4M
    benchmark_online_softmax(1 << 24);   // 16M
    
    return 0;
}

