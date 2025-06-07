#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

/*
    在CUDA中，warp是GPU调度和执行指令的最小单位, 它是指32个线程的一个组, 这32个线程会同步执行相同的指令；
    分支代价: 如果warp内线程执行不同分支, 会引发分支分化，导致性能下降；
    高效通信: cuda提供__shfl_*等warp原语,线程之间可以快速交换数据
    __shfl_xor_sync(mask, val, lane_mask)
    mask = 0xffffffff: 表示所有线程都参与(32个线程全参与)
    val: 每个线程的本地值
    lane_id: 一个warp有32个线程，编号是从0到31,这个编号是lane id
    lane_id ^ mask: 每个线程从编号land_id ^ mask的线程读取一个val值
*/
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

/*
    NUM_THREADS: 每个block的线程数
*/
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
        reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y,
                                                      int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    
    float4 reg_a = FLOAT4(a[idx]);
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
        reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    return val_f32;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = __half2float(sum_f16);
    __syncthreads();
    float sum = (lane < NUM_THREADS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_TREAHDS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_TREAHDS + tid;
    constexpr int NUM_WARPS = (NUM_TREAHDS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = sum_f32;
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    
    half2 reg_a = HALF2(a[idx]);
    half sum_f16  = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = sum_f32;
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    
    half2 reg_a = HALF2(a[idx]);
    half sum_f16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = __half2float(sum_f16);
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    
    __shared__ float reduce_smem[NUM_WARPS];
    half pack_a[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    const half z = __float2half(0.0f);
    half sum_f16 = z;
    
    for (int i = 0; i < 8; ++i)
        sum_f16 += (((idx + i) < N) ? pack_a[i] : z);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = __half2float(sum_f16);
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half pack_a[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);

    float sum_f32 = 0.0f;
    for (int i = 0; i < 8; i++)
        sum_f32 += (((idx + i) < N) ? __half2float(pack_a[i]) : 0.0f);
    
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f32 = warp_reduce_sum_f32<WARP_SIZE>(sum_f32);
    if (lane == 0)
        reduce_smem[warp] = sum_f32;
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(__nv_bfloat16 val) {
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val) {
    float val_f32 = __bfloat162float(val);
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    return val_f32;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16 *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
    
    __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
    if (lane == 0)
        reduce_smem[warp] = sum_bf16;
    __syncthreads();
    __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
    if (warp == 0)
        sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __bfloat162float(sum));
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16 *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    
    __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    if (lane == 0)
        reduce_smem[warp] = sum_f32;
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16 *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
    
    __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
    __nv_bfloat16 sum_bf16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
    if (lane == 0)
        reduce_smem[warp] = sum_bf16;
    __syncthreads();
    __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
    if (warp == 0)
        sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __bfloat162float(sum));
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16 *a, float *y, int  N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    
    __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
    __nv_bfloat16 sum_bf16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    if (lane == 0)
        reduce_smem[warp] = sum_f32;
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16 *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
    __nv_bfloat16 pack_a[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    const __nv_bfloat16 z = __float2bfloat16(0.0f);
    __nv_bfloat16 sum_bf16 = z;
    for (int i = 0; i < 8; i++)
        sum_bf16 += (((idx + i) < N) ? pack_a[i] : z);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
    if (lane == 0)
        reduce_smem[warp] = sum_bf16;
    __syncthreads();
    __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : z;
    if (warp == 0)
        sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __bfloat162float(sum));
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16 *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    __nv_bfloat16 pack_a[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    const __nv_bfloat16 z = __float2bfloat16(0.0f);
    
    __nv_bfloat16 sum_bf16 = z;
    for (int i = 0; i < 8; i++)
        sum_bf16 += (((idx + i) < N) ? pack_a[i] : z);
    
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    if (lane == 0)
        reduce_smem[warp] = sum_f32;
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) {
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    return val_f16;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val) {
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    return val_f16;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(__nv_fp8_storage_t *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];
    
    __nv_fp8_storage_t sum_f8 = (idx < N) ? a[idx] : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_f8);
    if (lane == 0)
        reduce_smem[warp] = sum_f16;
    __syncthreads();
    half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
    if (warp == 0)
        sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __half2float(sum));
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(__nv_fp8_storage_t *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];
    
    __nv_fp8_storage_t sum_f8 = (idx < N) ? a[idx] : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E5M2);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    half sum_f16 = warp_reduce_sum_fp8_e5m2_f16<WARP_SIZE>(sum_f8);
    if (lane == 0)
        reduce_smem[warp] = sum_f16;
    __syncthreads();
    half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
    if (warp == 0)
        sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __half2float(sum));
}

template <const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(__nv_fp8_storage_t *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];
    __nv_fp8_storage_t pack_a[16];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    
    half sum_f16 = __float2half(0.0f);
    for (int i = 0; i < 16; ++i)
        sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E4M3);
    
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = sum_f16;
    __syncthreads();
    half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
    if (warp == 0)
        sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __half2float(sum));
}

template <const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(__nv_fp8_storage_t *a,
                                                                 float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];
    __nv_fp8_storage_t pack_a[16];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    
    half sum_f16 = __float2half(0.0f);
    for (int i = 0; i < 16; i++)
        sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E5M2);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0)
        reduce_smem[warp] = sum_f16;
    __syncthreads();
    half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
    if (warp == 0)
        sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, __half2float(sum));
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
    int32_t val_i32 = static_cast<int32_t>(val);
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
    return val_i32;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(int8_t *a, int32_t *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ int32_t reduce_smem[NUM_WARPS];
    
    int8_t sum_i8 = (idx < N) ? a[idx] : 0;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(sum_i8);
    if (lane == 0)
        reduce_smem[warp] = sum_i32;
    __syncthreads();
    
    int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
    if (warp == 0)
        sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(int8_t *a, int32_t *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ int32_t reduce_smem[NUM_WARPS];
    int8_t pack_a[16];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    int32_t sum_i32 = 0;
    for (int i = 0; i < 16; i++)
        sum_i32 += (static_cast<int32_t>(pack_a[i]));
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_i32 = warp_reduce_sum_i32_i32<WARP_SIZE>(sum_i32);
    if (lane == 0)
        reduce_smem[warp] = sum_i32;
    __syncthreads();
    int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
    if (warp == 0)
        sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                        \
    m.def(STRINGFY(func), &func, STRINGFY(func));
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                        \
    if (((T).options().dtype() != (th_type))) {                      \
        std::cout << "Tensor Info:" << (T).options() << std::endl;  \
        throw std::runtime_error("values must be " #th_type);       \
    }

#define LAUNCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type,       \
                             out_type)                                      \
    block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)>          \
        <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),   \
                          reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,              \
                               n_elements, out_type)                                \
    const int NT = (K) / (n_elements);                                              \
    dim3 block(NT);                                                                 \
    dim3 grid((S));                                                                 \
    switch (NT) {                                                                   \
    case 32:                                                                        \
        LAUNCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)     \
        break;                                                                      \
    case 64:                                                                        \
        LAUNCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)     \
        break;                                                                      \
    case 128:                                                                       \
        LAUNCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)    \
        break;                                                                      \
    case 256:                                                                       \
        LAUNCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)    \
        break;                                                                      \
    case 512:                                                                       \
        LAUNCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)    \
        break;                                                                      \
    case 1024:                                                                      \
        LAUNCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)   \
        break;                                                                      \
    default:                                                                        \
        throw std::runtime_error(                                                   \
            "only support (K) / (n_elements): 32/64/128/256/512/1024"               \
        );                                                                          \
        break;                                                                      \
    }

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,              \
                             n_elements, out_type)                                      \
    torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(                      \
        torch::Tensor x) {                                                              \
            CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                      \
            auto y_th_type =                                                            \
                (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;            \
            auto options =                                                              \
                torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);        \
            auto y = torch::zeros({1}, options);                                        \
            const int ndim = x.dim();                                                   \
            if (ndim != 2) {                                                            \
                int N = 1;                                                              \
                for (int i = 0; i < ndim; ++i)                                          \
                    N *= x.size(i);                                                     \
                dim3 block(1024 / (n_elements));                                        \
                dim3 grid((N + 1024 - 1) / 1024);                                       \
                block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /            \
                                                                      (n_elements)>     \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),   \
                                      reinterpret_cast<out_type *>(y.data_ptr()), N);   \
            } else {                                                                    \
                const int S = x.size(0);                                                \
                const int K = x.size(1);                                                \
                const int N = S * K;                                                    \
                if ((K / (n_elements)) <= 1024) {                                       \
                    DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,      \
                                           n_elements, out_type)                        \
                } else {                                                                \
                    int N = 1;                                                          \
                    for (int i = 0; i < ndim; ++i)                                      \
                        N *= x.size(i);                                                 \
                    dim3 block(1024 / (n_elements));                                    \
                    dim3 grid((N + 1024 - 1) / 1024);                                   \
                    block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /     \
                                                                            (n_elements)>   \
                        <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),   \
                                          reinterpret_cast<out_type *>(y.data_ptr()), N);   \
                }                                                                           \
            }                                                                               \
            return y;                                                                       \
        }

TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, float, 1, float)
TORCH_BINDING_REDUCE(f32x4, f32, torch::kFloat32, float, 4, float)
TORCH_BINDING_REDUCE(f16, f16, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16, f32, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16x2, f16, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x2, f32, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x8_pack, f16, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(f16x8_pack, f32, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(bf16, bf16, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16, f32, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16x2, bf16, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x2, f32, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x8_pack, bf16, torch::kBFloat16, __nv_bfloat16, 8, float)
TORCH_BINDING_REDUCE(bf16x8_pack, f32, torch::kBFloat16, __nv_bfloat16, 8, float)
TORCH_BINDING_REDUCE(fp8_e4m3, f16, torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1, float)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16, torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2, f16, torch::kFloat8_e5m2, __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16, torch::kFloat8_e5m2, __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(i8, i32, torch::kInt8, int8_t, 1, int32_t)
TORCH_BINDING_REDUCE(i8x16_pack, i32, torch::kInt8, int8_t, 16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}
