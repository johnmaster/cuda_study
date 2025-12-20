#include "kernels.cuh"

void rand_init(float* h_data, size_t rows, size_t cols) {
    std::mt19937 gen(200);  // 固定种子，结果可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; i++) {
        h_data[i] = dist(gen);
    }
}

float* alloc_host(size_t n) {
    return new float[n];
}

float* alloc_device(size_t n) {
    float *p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

void free_host(float* p) {
    delete[] p;
}

void free_device(float* p) {
    CHECK_CUDA(cudaFree(p));
}

// Warp级别的shuffle规约
__device__ float warpReduceShuffle(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// 行规约 (Row Reduction)
// 输入: [rows, cols] 矩阵
// 输出: [rows] 向量，每个元素是对应行的和
// ============================================================================

/*
行规约 - 朴素版本
每个block处理一行，使用shared memory进行规约
*/
__global__ void reduce_rows_naive(float* device_out, const float* device_in, size_t rows, size_t cols) {
    extern __shared__ float sdata[];
    
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    
    if (row >= rows) return;
    
    // 每个线程累加多个元素
    float sum = 0.0f;
    for (size_t col = tid; col < cols; col += blockDim.x) {
        sum += device_in[row * cols + col];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // 归约shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        device_out[row] = sdata[0];
    }
}

/*
行规约 - 优化版本
使用warp shuffle，减少shared memory的使用
*/
__global__ void reduce_rows_optimized(float* device_out, const float* device_in, size_t rows, size_t cols) {
    __shared__ float sdata[32];  // 只需要存储每个warp的结果
    
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    
    if (row >= rows) return;
    
    // 每个线程累加多个元素
    float sum = 0.0f;
    for (size_t col = tid; col < cols; col += blockDim.x) {
        sum += device_in[row * cols + col];
    }
    
    // Warp级别的规约
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    // 最后一个warp归约所有warp的结果
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }
    
    if (tid == 0) {
        device_out[row] = sum;
    }
}

// ============================================================================
// 列规约 (Column Reduction)
// 输入: [rows, cols] 矩阵
// 输出: [cols] 向量，每个元素是对应列的和
// ============================================================================

/*
列规约 - 朴素版本
每个block处理一列，使用shared memory进行规约
*/
__global__ void reduce_cols_naive(float* device_out, const float* device_in, size_t rows, size_t cols) {
    extern __shared__ float sdata[];
    
    size_t col = blockIdx.x;
    size_t tid = threadIdx.x;
    
    if (col >= cols) return;
    
    // 每个线程累加多个元素
    float sum = 0.0f;
    for (size_t row = tid; row < rows; row += blockDim.x) {
        sum += device_in[row * cols + col];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // 归约shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        device_out[col] = sdata[0];
    }
}

/*
列规约 - 优化版本
使用warp shuffle，减少shared memory的使用
考虑内存访问的合并（coalescing）
*/
__global__ void reduce_cols_optimized(float* device_out, const float* device_in, size_t rows, size_t cols) {
    __shared__ float sdata[32];  // 只需要存储每个warp的结果
    
    size_t col = blockIdx.x;
    size_t tid = threadIdx.x;
    
    if (col >= cols) return;
    
    // 每个线程累加多个元素
    float sum = 0.0f;
    for (size_t row = tid; row < rows; row += blockDim.x) {
        sum += device_in[row * cols + col];
    }
    
    // Warp级别的规约
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    // 最后一个warp归约所有warp的结果
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }
    
    if (tid == 0) {
        device_out[col] = sum;
    }
}

// ============================================================================
// 全局规约 (Global Reduction)
// 输入: [rows, cols] 矩阵
// 输出: 单个标量值，所有元素的和
// 使用两阶段规约
// ============================================================================

/*
全局规约 - 第一阶段
每个block规约一部分数据
*/
__global__ void reduce_global_stage1(float* device_out, const float* device_in, size_t n) {
    __shared__ float sdata[32];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 每个线程处理两个元素
    float sum = 0.0f;
    if (idx < n)
        sum = device_in[idx];
    if (idx + blockDim.x < n)
        sum += device_in[idx + blockDim.x];
    
    // Warp级别的规约
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0)
        sdata[warp_id] = sum;
    __syncthreads();
    
    // 最后一个warp归约所有warp的结果
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }
    
    if (tid == 0)
        device_out[blockIdx.x] = sum;
}

/*
全局规约 - 第二阶段
归约第一阶段的结果（如果需要）
对于小规模的中间结果，可以用一个block完成
*/
__global__ void reduce_global_stage2(float* device_out, const float* device_in, size_t n) {
    __shared__ float sdata[32];
    
    size_t tid = threadIdx.x;
    
    // 每个线程累加多个元素
    float sum = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x) {
        sum += device_in[i];
    }
    
    // Warp级别的规约
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0)
        sdata[warp_id] = sum;
    __syncthreads();
    
    // 最后一个warp归约所有warp的结果
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }
    
    if (tid == 0)
        device_out[0] = sum;
}

// ============================================================================
// CPU验证函数
// ============================================================================

void verify_row_reduction(const float* input, const float* output, size_t rows, size_t cols) {
    const float epsilon = 1e-3f;
    bool all_correct = true;
    
    for (size_t i = 0; i < rows; i++) {
        float expected = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            expected += input[i * cols + j];
        }
        
        float diff = fabs(expected - output[i]);
        if (diff > epsilon) {
            printf("行 %zu 不匹配: 期望 %.6f, 实际 %.6f, 差值 %.6f\n", 
                   i, expected, output[i], diff);
            all_correct = false;
            if (!all_correct) break;  // 只显示第一个错误
        }
    }
    
    if (all_correct) {
        printf("✓ 行规约验证通过!\n");
    }
}

void verify_col_reduction(const float* input, const float* output, size_t rows, size_t cols) {
    const float epsilon = 1e-3f;
    bool all_correct = true;
    
    for (size_t j = 0; j < cols; j++) {
        float expected = 0.0f;
        for (size_t i = 0; i < rows; i++) {
            expected += input[i * cols + j];
        }
        
        float diff = fabs(expected - output[j]);
        if (diff > epsilon) {
            printf("列 %zu 不匹配: 期望 %.6f, 实际 %.6f, 差值 %.6f\n", 
                   j, expected, output[j], diff);
            all_correct = false;
            if (!all_correct) break;  // 只显示第一个错误
        }
    }
    
    if (all_correct) {
        printf("✓ 列规约验证通过!\n");
    }
}

void verify_global_reduction(const float* input, float output, size_t n) {
    const float epsilon = 5e-2f;  // 对于大规模数据，允许更大的浮点误差
    
    float expected = 0.0f;
    for (size_t i = 0; i < n; i++) {
        expected += input[i];
    }
    
    float diff = fabs(expected - output);
    if (diff > epsilon) {
        printf("全局规约不匹配: 期望 %.6f, 实际 %.6f, 差值 %.6f\n", 
               expected, output, diff);
    } else {
        printf("✓ 全局规约验证通过!\n");
    }
}

