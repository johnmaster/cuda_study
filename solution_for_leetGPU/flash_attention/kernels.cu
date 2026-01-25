/**
 * Flash Attention CUDA Implementation
 * 
 * 参考论文: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * 
 * 核心优化:
 * 1. Tiling: 分块加载 Q, K, V 到 shared memory
 * 2. Online Softmax: 在一次遍历中计算 softmax
 * 3. 不存储 N×N 的注意力矩阵
 */

#include "kernels.cuh"

// ============================================================================
// Helper Functions
// ============================================================================

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < n; i++) {
        h_data[i] = dist(gen);
    }
}

float* alloc_host(size_t n) {
    return new float[n];
}

float* alloc_device(size_t n) {
    float* p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

void free_host(float* p) {
    delete[] p;
}

void free_device(float* p) {
    CHECK_CUDA(cudaFree(p));
}

// ============================================================================
// Warp-level Reductions
// ============================================================================

__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);  // broadcast to all lanes
}

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);  // broadcast to all lanes
}

// ============================================================================
// Flash Attention Kernel (Single Head)
// ============================================================================

/**
 * Flash Attention Forward Kernel
 * 
 * 每个 block 处理一个 batch 的 Br 行 query
 * 
 * Grid: (ceil(N/Br), B)
 * Block: (Bc, Br) 或者 (32, 32)
 * 
 * Shared memory layout:
 * - sQ[Br][d]: Query block
 * - sK[Bc][d]: Key block
 * - sV[Bc][d]: Value block
 * - sS[Br][Bc]: Attention scores block
 */
template<int BR, int BC, int D>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,    // [B, N, d]
    const float* __restrict__ K,    // [B, N, d]
    const float* __restrict__ V,    // [B, N, d]
    float* __restrict__ O,          // [B, N, d]
    float* __restrict__ L,          // [B, N] - 用于存储 log-sum-exp (可选，用于 backward)
    float* __restrict__ M,          // [B, N] - 用于存储 max (可选，用于 backward)
    int N, int d
) {
    // Block indices
    int batch = blockIdx.y;
    int block_row = blockIdx.x;  // 处理第 block_row * BR 到 (block_row+1) * BR 行
    
    // Thread indices within block
    int tx = threadIdx.x;  // 0 to BC-1
    int ty = threadIdx.y;  // 0 to BR-1
    int tid = ty * BC + tx;
    
    // Base pointers for this batch
    const float* Q_batch = Q + batch * N * d;
    const float* K_batch = K + batch * N * d;
    const float* V_batch = V + batch * N * d;
    float* O_batch = O + batch * N * d;
    
    // Row index this thread is responsible for
    int row = block_row * BR + ty;
    
    // Shared memory
    extern __shared__ float smem[];
    float* sQ = smem;                           // [BR][D]
    float* sK = sQ + BR * D;                    // [BC][D]
    float* sV = sK + BC * D;                    // [BC][D]
    float* sS = sV + BC * D;                    // [BR][BC]
    
    // 每行的 running statistics (在寄存器中)
    float m_i = -INFINITY;  // running max
    float l_i = 0.0f;       // running sum of exp
    
    // Output accumulator (每行 d 维，存在寄存器中)
    // 由于 d 可能很大，我们分块处理
    float o_acc[4] = {0.0f};  // 每个线程负责 d 的一部分
    
    // 确定这个线程负责输出的哪些维度
    int d_per_thread = (d + BC - 1) / BC;
    
    // ========== Step 1: 加载 Q block 到 shared memory ==========
    // 每个线程加载多个元素
    for (int i = tid; i < BR * d; i += BR * BC) {
        int r = i / d;
        int c = i % d;
        int global_row = block_row * BR + r;
        if (global_row < N && c < d) {
            sQ[r * D + c] = Q_batch[global_row * d + c];
        } else {
            sQ[r * D + c] = 0.0f;
        }
    }
    __syncthreads();
    
    // ========== Step 2: 遍历 K, V 的每个块 ==========
    int num_kv_blocks = (N + BC - 1) / BC;
    
    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_start = j * BC;
        
        // 加载 K block
        for (int i = tid; i < BC * d; i += BR * BC) {
            int r = i / d;
            int c = i % d;
            int global_row = kv_start + r;
            if (global_row < N && c < d) {
                sK[r * D + c] = K_batch[global_row * d + c];
            } else {
                sK[r * D + c] = 0.0f;
            }
        }
        
        // 加载 V block
        for (int i = tid; i < BC * d; i += BR * BC) {
            int r = i / d;
            int c = i % d;
            int global_row = kv_start + r;
            if (global_row < N && c < d) {
                sV[r * D + c] = V_batch[global_row * d + c];
            } else {
                sV[r * D + c] = 0.0f;
            }
        }
        __syncthreads();
        
        // ========== Step 2a: 计算 S_ij = Q_i @ K_j^T / sqrt(d) ==========
        // 每个线程 (tx, ty) 计算 S[ty][tx]
        if (row < N) {
            float score = 0.0f;
            #pragma unroll
            for (int k = 0; k < D && k < d; k++) {
                score += sQ[ty * D + k] * sK[tx * D + k];
            }
            score *= rsqrtf((float)d);  // scale by 1/sqrt(d)
            
            // 边界检查
            int col = kv_start + tx;
            if (col >= N) {
                score = -INFINITY;
            }
            
            sS[ty * BC + tx] = score;
        }
        __syncthreads();
        
        // ========== Step 2b: 计算局部 max 和 更新 running max ==========
        if (row < N) {
            // 找这一行的局部 max
            float row_max = -INFINITY;
            for (int c = 0; c < BC; c++) {
                row_max = fmaxf(row_max, sS[ty * BC + c]);
            }
            
            // Online softmax 更新
            float m_ij = row_max;
            float m_new = fmaxf(m_i, m_ij);
            
            // ========== Step 2c: 计算 P_ij = exp(S_ij - m_new) 并累加 ==========
            float l_ij = 0.0f;
            for (int c = 0; c < BC; c++) {
                float p = expf(sS[ty * BC + c] - m_new);
                sS[ty * BC + c] = p;  // 复用 sS 存储 P
                l_ij += p;
            }
            
            // ========== Step 2d: 更新输出累加器 O_i ==========
            // O_i = O_i * exp(m_i - m_new) + P_ij @ V_j
            float scale = expf(m_i - m_new);
            
            // 更新 O 的每个维度
            for (int k_idx = 0; k_idx < d_per_thread && tx * d_per_thread + k_idx < d; k_idx++) {
                int k = tx * d_per_thread + k_idx;
                if (k < d) {
                    float pv = 0.0f;
                    for (int c = 0; c < BC; c++) {
                        pv += sS[ty * BC + c] * sV[c * D + k];
                    }
                    o_acc[k_idx] = o_acc[k_idx] * scale + pv;
                }
            }
            
            // ========== Step 2e: 更新 running statistics ==========
            l_i = l_i * scale + l_ij;
            m_i = m_new;
        }
        __syncthreads();
    }
    
    // ========== Step 3: 最终归一化并写回 ==========
    if (row < N) {
        float inv_l = 1.0f / l_i;
        
        for (int k_idx = 0; k_idx < d_per_thread && tx * d_per_thread + k_idx < d; k_idx++) {
            int k = tx * d_per_thread + k_idx;
            if (k < d) {
                O_batch[row * d + k] = o_acc[k_idx] * inv_l;
            }
        }
        
        // 可选：存储 L 和 M 用于 backward
        if (L != nullptr && tx == 0) {
            L[batch * N + row] = m_i + logf(l_i);
        }
        if (M != nullptr && tx == 0) {
            M[batch * N + row] = m_i;
        }
    }
}

// ============================================================================
// Simplified Flash Attention (更易理解的版本)
// ============================================================================

/**
 * 简化版 Flash Attention
 * 每个 block 处理一行 query
 * 更容易理解，但性能不如完整版
 */
__global__ void flash_attention_simple_kernel(
    const float* __restrict__ Q,    // [B, N, d]
    const float* __restrict__ K,    // [B, N, d]
    const float* __restrict__ V,    // [B, N, d]
    float* __restrict__ O,          // [B, N, d]
    int N, int d
) {
    // 每个 block 处理一行
    int batch = blockIdx.y;
    int row = blockIdx.x;
    
    if (row >= N) return;
    
    // Base pointers
    const float* q = Q + batch * N * d + row * d;  // [d]
    const float* K_batch = K + batch * N * d;       // [N, d]
    const float* V_batch = V + batch * N * d;       // [N, d]
    float* o = O + batch * N * d + row * d;         // [d]
    
    // Shared memory for K and V blocks
    extern __shared__ float smem[];
    float* sK = smem;                    // [Bc][d]
    float* sV = sK + Bc * d;             // [Bc][d]
    float* sq = sV + Bc * d;             // [d] - query cached
    
    int tx = threadIdx.x;
    
    // Load query to shared memory
    for (int i = tx; i < d; i += blockDim.x) {
        sq[i] = q[i];
    }
    __syncthreads();
    
    // Running statistics
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Output accumulator (each thread handles part of d)
    float o_local[8] = {0.0f};  // 每个线程最多处理 8 个维度
    int d_per_thread = (d + blockDim.x - 1) / blockDim.x;
    int d_start = tx * d_per_thread;
    int d_end = min(d_start + d_per_thread, d);
    
    // 遍历 K, V 的每个块
    int num_blocks = (N + Bc - 1) / Bc;
    
    for (int j = 0; j < num_blocks; j++) {
        int kv_start = j * Bc;
        int kv_end = min(kv_start + Bc, N);
        int block_size = kv_end - kv_start;
        
        // 加载 K block
        for (int i = tx; i < block_size * d; i += blockDim.x) {
            int r = i / d;
            int c = i % d;
            sK[r * d + c] = K_batch[(kv_start + r) * d + c];
        }
        
        // 加载 V block
        for (int i = tx; i < block_size * d; i += blockDim.x) {
            int r = i / d;
            int c = i % d;
            sV[r * d + c] = V_batch[(kv_start + r) * d + c];
        }
        __syncthreads();
        
        // 每个线程计算部分 scores 并进行 online softmax
        for (int col = 0; col < block_size; col++) {
            // 计算 score = q @ k[col] / sqrt(d)
            float score = 0.0f;
            for (int k = tx; k < d; k += blockDim.x) {
                score += sq[k] * sK[col * d + k];
            }
            // Warp reduce sum
            score = warpReduceSum(score);
            score *= rsqrtf((float)d);
            
            // 只有 lane 0 有正确的 score，需要 broadcast
            score = __shfl_sync(0xffffffff, score, 0);
            
            // Online softmax update
            float m_new = fmaxf(m_i, score);
            float p = expf(score - m_new);
            float scale = expf(m_i - m_new);
            
            // Update output accumulator
            for (int k = d_start; k < d_end; k++) {
                int idx = k - d_start;
                o_local[idx] = o_local[idx] * scale + p * sV[col * d + k];
            }
            
            // Update running statistics
            l_i = l_i * scale + p;
            m_i = m_new;
        }
        __syncthreads();
    }
    
    // 最终归一化并写回
    float inv_l = 1.0f / l_i;
    for (int k = d_start; k < d_end; k++) {
        int idx = k - d_start;
        o[k] = o_local[idx] * inv_l;
    }
}

// ============================================================================
// Standard Attention (for comparison)
// ============================================================================

__global__ void compute_qk_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N, int d
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        const float* q = Q + batch * N * d + row * d;
        const float* k = K + batch * N * d + col * d;
        
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += q[i] * k[i];
        }
        sum *= rsqrtf((float)d);
        
        S[batch * N * N + row * N + col] = sum;
    }
}

__global__ void softmax_rows_kernel(
    float* __restrict__ S,
    int N
) {
    int batch = blockIdx.y;
    int row = blockIdx.x;
    
    float* s_row = S + batch * N * N + row * N;
    
    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, s_row[i]);
    }
    max_val = warpReduceMax(max_val);
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float e = expf(s_row[i] - max_val);
        s_row[i] = e;
        sum += e;
    }
    sum = warpReduceSum(sum);
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s_row[i] *= inv_sum;
    }
}

__global__ void compute_sv_kernel(
    const float* __restrict__ S,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < d) {
        const float* s_row = S + batch * N * N + row * N;
        const float* V_batch = V + batch * N * d;
        
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += s_row[i] * V_batch[i * d + col];
        }
        
        O[batch * N * d + row * d + col] = sum;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d
) {
    // 使用简化版 kernel（更稳定）
    dim3 grid(N, B);
    dim3 block(32);  // 一个 warp
    
    size_t smem_size = (2 * Bc * d + d) * sizeof(float);
    
    flash_attention_simple_kernel<<<grid, block, smem_size>>>(Q, K, V, O, N, d);
}

void flash_attention_multihead(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d_model, int num_heads
) {
    int d_h = d_model / num_heads;
    
    // 对每个 head 调用 flash attention
    for (int h = 0; h < num_heads; h++) {
        // 创建指向当前 head 数据的指针（stride 访问）
        // 实际实现中应该用一个支持 stride 的 kernel
        // 这里简化为依次处理每个 head
        
        // 为当前 head 分配临时存储
        float *d_Q_h, *d_K_h, *d_V_h, *d_O_h;
        size_t head_size = B * N * d_h;
        
        CHECK_CUDA(cudaMalloc(&d_Q_h, head_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K_h, head_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V_h, head_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_O_h, head_size * sizeof(float)));
        
        // Copy strided data to contiguous (在实际实现中应该避免这个)
        // 这里为了简化先这样做
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                CHECK_CUDA(cudaMemcpy(
                    d_Q_h + b * N * d_h + n * d_h,
                    Q + b * N * d_model + n * d_model + h * d_h,
                    d_h * sizeof(float),
                    cudaMemcpyDeviceToDevice
                ));
                CHECK_CUDA(cudaMemcpy(
                    d_K_h + b * N * d_h + n * d_h,
                    K + b * N * d_model + n * d_model + h * d_h,
                    d_h * sizeof(float),
                    cudaMemcpyDeviceToDevice
                ));
                CHECK_CUDA(cudaMemcpy(
                    d_V_h + b * N * d_h + n * d_h,
                    V + b * N * d_model + n * d_model + h * d_h,
                    d_h * sizeof(float),
                    cudaMemcpyDeviceToDevice
                ));
            }
        }
        
        // Run flash attention on this head
        flash_attention_forward(d_Q_h, d_K_h, d_V_h, d_O_h, B, N, d_h);
        
        // Copy result back
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                CHECK_CUDA(cudaMemcpy(
                    O + b * N * d_model + n * d_model + h * d_h,
                    d_O_h + b * N * d_h + n * d_h,
                    d_h * sizeof(float),
                    cudaMemcpyDeviceToDevice
                ));
            }
        }
        
        CHECK_CUDA(cudaFree(d_Q_h));
        CHECK_CUDA(cudaFree(d_K_h));
        CHECK_CUDA(cudaFree(d_V_h));
        CHECK_CUDA(cudaFree(d_O_h));
    }
}

void standard_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d
) {
    // 分配 S 矩阵
    float* d_S;
    CHECK_CUDA(cudaMalloc(&d_S, B * N * N * sizeof(float)));
    
    // Step 1: S = Q @ K^T / sqrt(d)
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16, B);
        compute_qk_kernel<<<grid, block>>>(Q, K, d_S, N, d);
    }
    
    // Step 2: Softmax rows
    {
        dim3 grid(N, B);
        dim3 block(32);
        softmax_rows_kernel<<<grid, block>>>(d_S, N);
    }
    
    // Step 3: O = S @ V
    {
        dim3 block(16, 16);
        dim3 grid((d + 15) / 16, (N + 15) / 16, B);
        compute_sv_kernel<<<grid, block>>>(d_S, V, O, N, d);
    }
    
    CHECK_CUDA(cudaFree(d_S));
}

