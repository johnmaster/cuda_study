#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(200);  // 固定种子，结果可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
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

/*
sdata = [0, 1, 2, 3, 4, 5, 6, 7]
s = 1
tid = 0 sdata[0] += sdata[1] = 1
tid = 2 sdata[2] += sdata[3] = 5
tid = 4 sdata[4] += sdata[5] = 9
tid = 6 sdata[6] += sdata[7] = 13

sdata = [1, 1, 5, 3, 9, 5, 13, 7]
s = 2
tid = 0 sdata[0] += sdata[2] = 6
tid = 4 sdata[4] += sdata[6] = 22

sdata = [6, 1, 5, 3, 22, 5, 13, 7]
s = 4
tid = 0 sdata[0] += sdata[4] = 28
*/
__global__ void reduce_naive(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? device_in[idx] : 0.0f;
    __syncthreads();
    
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        device_out[blockIdx.x] = sdata[0];
    }
}

/*
sdata = [0, 1, 2, 3, 4, 5, 6, 7]
s = 1
tid = 0 index = 0 sdata[0] += sdata[1] = 1
tid = 1 index = 2 sdata[2] += sdata[3] = 5
tid = 2 index = 4 sdata[4] += sdata[5] = 9
tid = 3 index = 6 sdata[6] += sdata[7] = 13

sdata = [1, 1, 5, 3, 9, 5, 13, 7]
s = 2
tid = 0 index = 0 sdata[0] += sdata[2] = 6
tid = 1 index = 4 sdata[4] += sdata[6] = 22

sdata = [6, 1, 5, 3, 22, 5, 13, 7]
s = 4
tid = 0 index = 0 sdata[0] += sdata[4] = 28
*/
__global__ void reduce_no_divergence(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? device_in[idx] : 0.0f;
    __syncthreads();
    
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        device_out[blockIdx.x] = sdata[0];
}

/*
sdata = [0, 1, 2, 3, 4, 5, 6, 7]
s = 4
tid = 0 sdata[0] += sdata[4] = 4
tid = 1 sdata[1] += sdata[5] = 6
tid = 2 sdata[2] += sdata[6] = 8
tid = 3 sdata[3] += sdata[7] = 10

sdata = [4, 6, 8, 10, 4, 5, 6, 7]
s = 2
tid = 0 sdata[0] += sdata[2] = 12
tid = 1 sdata[1] += sdata[3] = 16

sdata = [12, 16, 8, 10, 4, 5, 6, 7]
s = 1
tid = 0 sdata[0] += sdata[1] = 28
*/
__global__ void reduce_sequential(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? device_in[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        device_out[blockIdx.x] = sdata[0];
}


__global__ void reduce_first_add(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    //first load execute one addition
    float sum = 0.0f;
    if (idx < n)
        sum = device_in[idx];
    if (idx + blockDim.x < n)
        sum += device_in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        device_out[blockIdx.x] = sdata[0];
}

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_unroll_last_warp(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n)
        sum = device_in[idx];
    if (idx + blockDim.x < n)
        sum += device_in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(sdata, tid);

    if (tid == 0)
        device_out[blockIdx.x] = sdata[0];
}

/*
__shfl_down_sync: every thread get value from thread + offset
offset = 16
thread0 : thread0 + thread16
thread1 : thread1 + thread17
...
thread15: thread15 + thread31

offset = 8
thread0: thread0 + thread8
thread1: thread1 + thread9
...
thread7: thread7 + thread15

offset = 4
thread0: thread0 + thread4
thread1: thread1 + thread5
...
thread3: thread3 + thread7

offset = 2
thread0: thread0 + thread2
thread1: thread1 + thread3

offset = 1
thread0: thread0 + thread1
*/
__device__ float warpReduceShuffle(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
__global__ void reduce_warp_shuffle(float* device_out, float*device_in, size_t n) {
    __shared__ float sdata[32];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (idx < n)
        sum = device_in[idx];
    if (idx + blockDim.x < n)
        sum += device_in[idx + blockDim.x];

    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0)
        sdata[warp_id] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }

    if (tid == 0)
        device_out[blockIdx.x] = sum;
}

/*
V8: 向量化读取 (Vectorized Load) - 修正版
使用 float4 一次读取 4 个 float (128 bits)

正确的 coalesced 访问模式:
- warp 内 32 线程同时读取 32 个 float4
- 每个线程读的 float4 地址相邻
- 一个 warp 一次读取 32 * 4 = 128 个 float

每个 block 处理: BLOCK_SIZE * 4 个元素
*/
__global__ void reduce_vectorized(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[32];  // 只需要存 warp 结果
    
    size_t tid = threadIdx.x;
    // 正确的向量化索引: 相邻线程读取相邻的 float4
    // blockIdx.x * blockDim.x * 4 是 block 的起始位置
    // tid * 4 是线程在 block 内的偏移（每线程 4 个 float）
    size_t base = blockIdx.x * blockDim.x * 4;
    size_t idx = base + tid;  // 先算 float 索引
    
    float sum = 0.0f;
    
    // 每个线程处理 4 个 strided 的元素（避免非对齐的 float4 读取）
    // 这样 warp 内是 coalesced 的
    if (idx < n) sum += device_in[idx];
    if (idx + blockDim.x < n) sum += device_in[idx + blockDim.x];
    if (idx + blockDim.x * 2 < n) sum += device_in[idx + blockDim.x * 2];
    if (idx + blockDim.x * 3 < n) sum += device_in[idx + blockDim.x * 3];
    
    // Warp shuffle 归约
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0)
        sdata[warp_id] = sum;
    __syncthreads();
    
    // 最后一个 warp 归约所有 warp 的结果
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }
    
    if (tid == 0)
        device_out[blockIdx.x] = sum;
}

/*
V9: 真正的 float4 向量化（要求地址 16-byte 对齐）
*/
__global__ void reduce_float4(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[32];
    
    size_t tid = threadIdx.x;
    // 转换为 float4 指针，要求输入地址 16-byte 对齐
    float4* device_in4 = reinterpret_cast<float4*>(device_in);
    size_t n4 = n / 4;  // float4 的数量
    
    // 每个 block 处理 blockDim.x 个 float4
    size_t idx = blockIdx.x * blockDim.x + tid;
    
    float sum = 0.0f;
    
    if (idx < n4) {
        float4 val = device_in4[idx];
        sum = val.x + val.y + val.z + val.w;
    }
    
    // Warp shuffle 归约
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0)
        sdata[warp_id] = sum;
    __syncthreads();
    
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }
    
    if (tid == 0)
        device_out[blockIdx.x] = sum;
}

__global__ void reduce_float4x2(float* device_out, float* device_in, size_t n) {
    __shared__ float sdata[32];

    size_t tid = threadIdx.x;
    float4* device_in4 = reinterpret_cast<float4*>(device_in);
    size_t n4 = n / 4;
    
    //every thread load two float4 elements
    size_t idx = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    
    if (idx < n4) {
        float4 val = device_in4[idx];
        sum += val.x + val.y + val.z + val.w;
    }
    
    if (idx + blockDim.x < n4) {
        float4 val = device_in4[idx + blockDim.x];
        sum += val.x + val.y + val.z + val.w;
    }

    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceShuffle(sum);
    
    if (lane == 0)
        sdata[warp_id] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceShuffle(sum);
    }

    if (tid == 0)
        device_out[blockIdx.x] = sum;
}