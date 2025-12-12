#include "kernels.cuh"
#include <random>

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(200);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
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

/*
 * Blelloch Scan - 单 block 内的前缀和
 * 
 * Up-sweep 示例 (8个元素):
 *   原始: [0, 1, 2, 3, 4, 5, 6, 7]
 *   d=4:  [0, 0+1, 2, 2+3, 4, 4+5, 6, 6+7] = [0,1,2,5,4,9,6,13]
 *   d=2:  [0, 1, 2, 1+5, 4, 9, 6, 9+13]    = [0,1,2,6,4,9,6,22]
 *   d=1:  [0, 1, 2, 6, 4, 9, 6, 6+22]      = [0,1,2,6,4,9,6,28]
 *   最后位置 = 总和28
 * 
 * Down-sweep 示例:
 *   设置: last = 0                          = [0,1,2,6,4,9,6,0]
 *   d=1:  swap(idx3,idx7), idx7+=old_idx3   = [0,1,2,0,4,9,6,6]
 *   d=2:  对应位置 swap+add                  = [0,0,2,1,4,6,6,15]
 *   d=4:  对应位置 swap+add                  = [0,0,1,3,6,10,15,21]
 *   结果是 exclusive scan，加上原始值得到 inclusive scan
 */
__global__ void block_scan_inclusive(float* d_output, float* d_input, size_t n) {
    __shared__ float temp[ELEMENTS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    // 每个线程加载2个元素到共享内存
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int global_ai = block_offset + ai;
    int global_bi = block_offset + bi;
    
    // 保存原始输入值（后面转换 exclusive→inclusive 需要）
    float orig_ai = (global_ai < n) ? d_input[global_ai] : 0.0f;
    float orig_bi = (global_bi < n) ? d_input[global_bi] : 0.0f;
    
    temp[ai] = orig_ai;
    temp[bi] = orig_bi;
    
    // ========== Up-sweep (Reduce) 阶段 ==========
    // 构建部分和的树
    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            // ai_idx: 当前线程的左子节点索引
            // bi_idx: 当前线程的右子节点索引
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }
    
    // 保存 block 总和，然后清零最后一个元素
    __syncthreads();
    float block_sum = temp[ELEMENTS_PER_BLOCK - 1];
    if (tid == 0) {
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    
    // ========== Down-sweep 阶段 ==========
    // 从树根向下传播，构建扫描结果
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            float t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    __syncthreads();
    
    // ========== Exclusive → Inclusive 转换 ==========
    // inclusive[i] = exclusive[i] + input[i]
    if (global_ai < n) {
        d_output[global_ai] = temp[ai] + orig_ai;
    }
    if (global_bi < n) {
        d_output[global_bi] = temp[bi] + orig_bi;
    }
    
    // 在输出的最后一个位置存储 block 总和（用于多 block 情况）
    // 实际上我们需要单独存储 block sums，这里简化处理
}

// 带 block_sums 输出的版本
__global__ void block_scan_with_sums(float* d_output, float* d_input, float* d_block_sums, size_t n) {
    __shared__ float temp[ELEMENTS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int global_ai = block_offset + ai;
    int global_bi = block_offset + bi;
    
    float orig_ai = (global_ai < n) ? d_input[global_ai] : 0.0f;
    float orig_bi = (global_bi < n) ? d_input[global_bi] : 0.0f;
    
    temp[ai] = orig_ai;
    temp[bi] = orig_bi;
    
    // Up-sweep
    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }
    
    __syncthreads();
    // 保存这个 block 的总和
    if (tid == 0) {
        d_block_sums[blockIdx.x] = temp[ELEMENTS_PER_BLOCK - 1];
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    
    // Down-sweep
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            float t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    __syncthreads();
    
    // Exclusive → Inclusive
    if (global_ai < n) {
        d_output[global_ai] = temp[ai] + orig_ai;
    }
    if (global_bi < n) {
        d_output[global_bi] = temp[bi] + orig_bi;
    }
}

// 将前面 block 的前缀和加到当前 block 的每个元素上
__global__ void add_block_sums(float* d_output, float* d_block_prefix, size_t n) {
    if (blockIdx.x == 0) return;  // 第一个 block 不需要加
    
    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    float prefix = d_block_prefix[blockIdx.x - 1];  // 前一个 block 的累积和
    
    int idx1 = block_offset + threadIdx.x;
    int idx2 = block_offset + threadIdx.x + BLOCK_SIZE;
    
    if (idx1 < n) {
        d_output[idx1] += prefix;
    }
    if (idx2 < n) {
        d_output[idx2] += prefix;
    }
}

/*
 * 递归处理大数组的前缀和
 * 
 * 思路:
 * 1. 每个 block 独立计算局部前缀和，同时输出该 block 的总和
 * 2. 对所有 block 总和递归做前缀和
 * 3. 把前缀结果加回到每个 block 的元素上
 */
void prefix_sum_recursive(float* d_input, float* d_output, size_t n) {
    int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    if (num_blocks == 1) {
        // 基础情况: 单个 block 可以处理
        block_scan_inclusive<<<1, BLOCK_SIZE>>>(d_output, d_input, n);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    // 分配存储 block 总和的空间
    float* d_block_sums = alloc_device(num_blocks);
    float* d_block_prefix = alloc_device(num_blocks);

    // Step 1: 每个 block 计算局部前缀和 + 输出 block 总和
    // d_block_sums: 每个 block 的总和
    block_scan_with_sums<<<num_blocks, BLOCK_SIZE>>>(d_output, d_input, d_block_sums, n);
    CHECK_CUDA(cudaGetLastError());

    // Step 2: 递归计算 block 总和的前缀和
    // d_block_prefix: 每个 block 的总和的前缀和
    prefix_sum_recursive(d_block_sums, d_block_prefix, num_blocks);
    
    // Step 3: 把前面 blocks 的累积和加到当前 block
    add_block_sums<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_prefix, n);
    CHECK_CUDA(cudaGetLastError());

    free_device(d_block_sums);
    free_device(d_block_prefix);
}

void solve(float* d_input, float* d_output, size_t n) {
    if (n == 0) return;

    prefix_sum_recursive(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());
}
