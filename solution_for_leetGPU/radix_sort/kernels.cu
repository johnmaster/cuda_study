#include "kernels.cuh"

void rand_init(uint32_t* h_data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (size_t i = 0; i < n; i++) {
        h_data[i] = dist(gen);
    }
}
uint32_t* alloc_host(size_t n) {
    return new uint32_t[n];
}
uint32_t* alloc_device(size_t n) {
    uint32_t* p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(uint32_t)));
    return p;
}
void free_host(uint32_t* p) {
    delete[] p;
}
void free_device(uint32_t* p) {
    CHECK_CUDA(cudaFree(p));
}

__device__ __forceinline__ uint32_t extract_digit(uint32_t value, int bit_offset) {
    return (value >> bit_offset) & (RADIX - 1);
}

// ============ 简单易懂版本 ============

// Kernel 1: 统计直方图 + 为每个元素分配桶内 rank
__global__ void histogram_and_rank_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ ranks,              // 每个元素的桶内序号
    uint32_t* __restrict__ block_histograms,   // [num_blocks][RADIX]
    size_t n,
    int bit_offset,
    size_t elements_per_block
) {
    __shared__ uint32_t local_hist[RADIX];
    
    // 初始化直方图
    if (threadIdx.x < RADIX) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    size_t block_start = blockIdx.x * elements_per_block;
    size_t block_end = min(block_start + elements_per_block, n);
    
    // 单线程按顺序分配 rank（保证稳定性）
    if (threadIdx.x == 0) {
        for (size_t i = block_start; i < block_end; i++) {
            uint32_t digit = extract_digit(input[i], bit_offset);
            ranks[i] = local_hist[digit]++;
        }
    }
    __syncthreads();
    
    // 写入全局直方图
    if (threadIdx.x < RADIX) {
        block_histograms[blockIdx.x * RADIX + threadIdx.x] = local_hist[threadIdx.x];
    }
}

// Kernel 2: 计算全局前缀和
__global__ void compute_offsets_kernel(
    uint32_t* __restrict__ block_histograms,  // [num_blocks][RADIX]
    int num_blocks
) {
    __shared__ uint32_t digit_totals[RADIX];    // every digit's total count in all blocks
    __shared__ uint32_t digit_offsets[RADIX];   // every digit's global offset in all blocks
    
    int digit = threadIdx.x;
    if (digit >= RADIX) return;
    
    // Step 1: 每个 digit 在 block 间做 exclusive prefix sum
    uint32_t running_sum = 0;
    for (int b = 0; b < num_blocks; b++) {
        uint32_t count = block_histograms[b * RADIX + digit];
        block_histograms[b * RADIX + digit] = running_sum;
        running_sum += count;
    }
    digit_totals[digit] = running_sum;
    __syncthreads();
    
    // Step 2: 对 digit 做 exclusive prefix sum（串行，简单）
    if (threadIdx.x == 0) {
        digit_offsets[0] = 0;
        for (int d = 1; d < RADIX; d++) {
            digit_offsets[d] = digit_offsets[d-1] + digit_totals[d-1];
        }
    }
    __syncthreads();
    
    // Step 3: 把 digit 全局偏移加到每个 block 的偏移上
    uint32_t offset = digit_offsets[digit];
    for (int b = 0; b < num_blocks; b++) {
        block_histograms[b * RADIX + digit] += offset;
    }
}

// Kernel 3: 并行写回
__global__ void scatter_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ ranks,
    const uint32_t* __restrict__ block_offsets,
    size_t n,
    int bit_offset,
    size_t elements_per_block
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint32_t value = input[idx];
    uint32_t digit = extract_digit(value, bit_offset);
    size_t sort_block = idx / elements_per_block;
    
    // 位置 = block 的 digit 全局起点 + 元素的桶内 rank
    uint32_t pos = block_offsets[sort_block * RADIX + digit] + ranks[idx];
    output[pos] = value;
}

void radix_sort_gpu(uint32_t* d_input, uint32_t* d_output, size_t n) {
    // 每个 block 处理约 256 元素
    int num_sort_blocks = min((int)((n + 255) / 256), 4096);
    size_t elements_per_block = (n + num_sort_blocks - 1) / num_sort_blocks;
    
    int scatter_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 分配临时空间
    // @d_ranks: every element's rank in the bucket
    // @d_block_histograms: every digit's total count in all blocks
    uint32_t* d_temp;
    uint32_t* d_ranks;
    uint32_t* d_block_histograms;
    CHECK_CUDA(cudaMalloc(&d_temp, n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_ranks, n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_block_histograms, num_sort_blocks * RADIX * sizeof(uint32_t)));
    
    uint32_t* src = d_input;
    uint32_t* dst = d_temp;
    
    // 8 轮排序（32 位 / 4 位）
    for (int bit = 0; bit < 32; bit += RADIX_BITS) {
        // Step 1: 统计直方图 + 分配 rank
        histogram_and_rank_kernel<<<num_sort_blocks, BLOCK_SIZE>>>(
            src, d_ranks, d_block_histograms, n, bit, elements_per_block
        );
        
        // Step 2: 计算全局偏移
        compute_offsets_kernel<<<1, RADIX>>>(d_block_histograms, num_sort_blocks);
        
        // Step 3: 并行写回
        scatter_kernel<<<scatter_blocks, BLOCK_SIZE>>>(
            src, dst, d_ranks, d_block_histograms, n, bit, elements_per_block
        );
        
        // 交换 src 和 dst
        uint32_t* tmp = src;
        src = dst;
        dst = tmp;
    }
    
    // 确保结果在 d_output
    if (src != d_output) {
        CHECK_CUDA(cudaMemcpy(d_output, src, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    }
    
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_ranks));
    CHECK_CUDA(cudaFree(d_block_histograms));
}