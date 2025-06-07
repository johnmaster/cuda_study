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

__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0f;
        for (int k = 0; k < K; k++)
            psum += a[m * K + k] * b[k * N + n];
        c[m * N + n] = psum;
    }
}

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ floatd s_a[BM][BK], s_b[BK][BN];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;
    
    int load_smem_a_m = tid / 32;
    int load_smem_a_k = tid % 32;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    
    float sum = 0.f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_gmem_a_m][load_gmem_a_k] = a[load_gmem_a_addr];
        int load_gmem_b_k = bk * BK + load_gmem_b_n;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = load_gmem_c_m * N + load_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32 * 4;
    int load_gmem_a_m = by * BM + load_gmem_a_m;
    int load_gmem_b_n = bx * BN + load_gmem_b_n;
    
    float r_c[TM][TN] = {0.0};
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(s_b[load_gmem_b_k][load_gmem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
        __syncthreads();
        
        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    int comp_smem_a_m = ty * TM + m;
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }
    
    for (int m = 0; m < TM; ++m) {
        int store_gmem_c_m = by * BM + ty * TM + m;
        for (int n = 0; n < TN; n += 4) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c,
                                                      const int M, const int N, const int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    __shared__ float s_a[BK][BM + OFFSET];
    __shared__ float s_b[BK][BN + OFFSET];
    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void sgemm_t_8x8_sliced_k_f32x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32);
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_t_8x8_sliced_k_f32_4_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_sliced_k_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_sliced_k_f32_kernel<BM, BN, BK>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 32;
    constexpr int BN = 32;
    
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    sgemm_naive_f32_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(a.data_ptr()),
        reinterpret_cast<float *>(b.data_ptr()),
        reinterpret_cast<float *>(c.data_ptr()), M, N, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA Cores
  TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_sliced_k_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_offset)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async)
  // cuBLAS Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(sgemm_cublas)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_cublas_tf32)
  // WMMA API Tensor Cores, stage, thread block swizzle, dsmem
  TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages)
  TORCH_BINDING_COMMON_EXTENSION(
      sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem)
}