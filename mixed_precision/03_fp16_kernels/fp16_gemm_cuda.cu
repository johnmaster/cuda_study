/**
 * FP16 vs FP32 GEMM CUDA Kernel 对比
 *
 * 展示:
 *   1. FP32 naive GEMM  — 基线
 *   2. FP16 naive GEMM  — 直接用 half, 2x 带宽节省
 *   3. FP16 WMMA GEMM   — 利用 Tensor Core, 真正的加速来源
 *
 * 编译:
 *   nvcc -O3 -arch=sm_80 -o fp16_gemm_cuda fp16_gemm_cuda.cu -lcudart
 *
 * 运行:
 *   ./fp16_gemm_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(cmd) do {                                       \
    cudaError_t e = cmd;                                           \
    if (e != cudaSuccess) {                                        \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
               cudaGetErrorString(e));                             \
        exit(1);                                                   \
    }                                                              \
} while(0)

// ============================================================================
// Kernel 1: FP32 Naive GEMM
// ============================================================================

__global__ void gemm_fp32_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 2: FP16 Naive GEMM (不用 Tensor Core)
// ============================================================================

__global__ void gemm_fp16_kernel(const half* A, const half* B, half* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

// ============================================================================
// Kernel 3: FP16 WMMA GEMM (Tensor Core)
// ============================================================================

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void gemm_wmma_kernel(const half* A, const half* B, float* C,
                                  int M, int N, int K) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;

    if (warpM >= M || warpN >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}

// ============================================================================
// Benchmark helper
// ============================================================================

float benchmark_kernel(void (*run)(void*), void* ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) run(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; i++) run(ctx);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

// ============================================================================
// Main
// ============================================================================

struct GemmCtx {
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    float *d_C_wmma;
    int M, N, K;
};

void run_fp32(void* p) {
    GemmCtx* ctx = (GemmCtx*)p;
    dim3 block(16, 16);
    dim3 grid((ctx->N + 15) / 16, (ctx->M + 15) / 16);
    gemm_fp32_kernel<<<grid, block>>>(
        ctx->d_A_fp32, ctx->d_B_fp32, ctx->d_C_fp32, ctx->M, ctx->N, ctx->K
    );
}

void run_fp16(void* p) {
    GemmCtx* ctx = (GemmCtx*)p;
    dim3 block(16, 16);
    dim3 grid((ctx->N + 15) / 16, (ctx->M + 15) / 16);
    gemm_fp16_kernel<<<grid, block>>>(
        ctx->d_A_fp16, ctx->d_B_fp16, ctx->d_C_fp16, ctx->M, ctx->N, ctx->K
    );
}

void run_wmma(void* p) {
    GemmCtx* ctx = (GemmCtx*)p;
    dim3 block(128, 4);
    dim3 grid((ctx->N + (WMMA_N * 4) - 1) / (WMMA_N * 4),
              (ctx->M + (WMMA_M * 1) - 1) / (WMMA_M * 1));
    gemm_wmma_kernel<<<grid, block>>>(
        ctx->d_A_fp16, ctx->d_B_fp16, ctx->d_C_wmma, ctx->M, ctx->N, ctx->K
    );
}

int main() {
    printf("FP32 vs FP16 vs WMMA (Tensor Core) GEMM\n");
    printf("========================================\n\n");

    int sizes[][3] = {{512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}};
    int num_sizes = 3;

    printf("  %15s  %10s  %10s  %10s\n", "Size", "FP32", "FP16", "WMMA(TC)");
    printf("  %15s  %10s  %10s  %10s\n",
           "───────────────", "──────────", "──────────", "──────────");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];

        GemmCtx ctx;
        ctx.M = M; ctx.N = N; ctx.K = K;

        CUDA_CHECK(cudaMalloc(&ctx.d_A_fp32, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_B_fp32, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_C_fp32, M * N * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&ctx.d_A_fp16, M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&ctx.d_B_fp16, K * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&ctx.d_C_fp16, M * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&ctx.d_C_wmma, M * N * sizeof(float)));

        float ms_fp32 = benchmark_kernel(run_fp32, &ctx, 5, 20);
        float ms_fp16 = benchmark_kernel(run_fp16, &ctx, 5, 20);
        float ms_wmma = benchmark_kernel(run_wmma, &ctx, 5, 20);

        printf("  %4dx%4dx%4d  %7.2f ms  %7.2f ms  %7.2f ms\n",
               M, N, K, ms_fp32, ms_fp16, ms_wmma);

        CUDA_CHECK(cudaFree(ctx.d_A_fp32));
        CUDA_CHECK(cudaFree(ctx.d_B_fp32));
        CUDA_CHECK(cudaFree(ctx.d_C_fp32));
        CUDA_CHECK(cudaFree(ctx.d_A_fp16));
        CUDA_CHECK(cudaFree(ctx.d_B_fp16));
        CUDA_CHECK(cudaFree(ctx.d_C_fp16));
        CUDA_CHECK(cudaFree(ctx.d_C_wmma));
    }

    printf("\n");
    printf("  FP16 naive: 节省一半带宽, 但不用 Tensor Core → 加速有限\n");
    printf("  WMMA:       用 Tensor Core → 真正的大幅加速\n");
    printf("  AMP 的加速来源: autocast 把 GEMM 转成 FP16 → cuBLAS 自动用 Tensor Core\n");

    return 0;
}
