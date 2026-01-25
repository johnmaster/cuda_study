#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// WMMA 分块尺寸
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// 版本1: 基础 WMMA 实现
// 每个 warp 计算一个 16x16 的输出块
// ============================================================================
__global__ void wmma_gemm_v1_naive(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
   // 计算当前 warp 负责的输出块位置
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
 
    // 边界检查
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // 声明 fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0.0f);

    // 沿 K 维度累加
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // 加载 A 和 B 的分片
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        // 矩阵乘加
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 存储结果
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}

// ============================================================================
// 版本2: 使用共享内存优化
// 将全局内存数据缓存到共享内存，减少全局内存访问
// ============================================================================
template<int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, int WARP_ROW_TILES, int WARP_COL_TILES>
__global__ void wmma_gemm_v2_shared(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 每个 block 处理的输出块大小
    constexpr int BLOCK_ROW_TILES = BLOCK_ROW_WARPS * WARP_ROW_TILES;
    constexpr int BLOCK_COL_TILES = BLOCK_COL_WARPS * WARP_COL_TILES;
    constexpr int BLOCK_M = BLOCK_ROW_TILES * WMMA_M;  // 64
    constexpr int BLOCK_N = BLOCK_COL_TILES * WMMA_N;  // 64
    constexpr int BLOCK_K = 32;

    // 共享内存
    __shared__ half As[BLOCK_M][BLOCK_K];
    __shared__ half Bs[BLOCK_K][BLOCK_N];

    // 线程和 warp 索引
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = warpId / BLOCK_COL_WARPS;
    int warpCol = warpId % BLOCK_COL_WARPS;

    // 块的起始位置
    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;

    // 声明累加器 fragments (每个 warp 处理多个 tile)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> 
        acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    // 沿 K 维度分块
    for (int k = 0; k < K; k += BLOCK_K) {
        // 协作加载 A 到共享内存
        // 64x32 = 2048 elements, 128 threads, every thread process 16 elements
        for (int loadIdx = threadIdx.x; loadIdx < BLOCK_M * BLOCK_K; loadIdx += blockDim.x) {
            int row = loadIdx / BLOCK_K;    // row number
            int col = loadIdx % BLOCK_K;    // col number
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
            As[row][col] = (globalRow < M && globalCol < K) ? 
                           A[globalRow * K + globalCol] : __float2half(0.0f);
        }

        // 协作加载 B 到共享内存
        for (int loadIdx = threadIdx.x; loadIdx < BLOCK_K * BLOCK_N; loadIdx += blockDim.x) {
            int row = loadIdx / BLOCK_N;
            int col = loadIdx % BLOCK_N;
            int globalRow = k + row;
            int globalCol = blockColStart + col;
            Bs[row][col] = (globalRow < K && globalCol < N) ? 
                           B[globalRow * N + globalCol] : __float2half(0.0f);
        }

        __syncthreads();

        // 从共享内存加载并计算
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_ROW_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_COL_TILES];

            // 加载 A fragments
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {  // row directions process 2 tile
                int row = warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[row][kk], BLOCK_K);
            }

            // 加载 B fragments
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++) {  // col directions process 2 tile
                int col = warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][col], BLOCK_N);
            }

            // 执行矩阵乘加
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++) {
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // 存储结果
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            int cRow = blockRowStart + warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
            int cCol = blockColStart + warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
            if (cRow < M && cCol < N) {
                wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// ============================================================================
// 版本3: 双缓冲 + 共享内存优化
// 使用双缓冲技术隐藏全局内存延迟
// ============================================================================
template<int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, int WARP_ROW_TILES, int WARP_COL_TILES>
__global__ void wmma_gemm_v3_double_buffer(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int BLOCK_ROW_TILES = BLOCK_ROW_WARPS * WARP_ROW_TILES;
    constexpr int BLOCK_COL_TILES = BLOCK_COL_WARPS * WARP_COL_TILES;
    constexpr int BLOCK_M = BLOCK_ROW_TILES * WMMA_M;
    constexpr int BLOCK_N = BLOCK_COL_TILES * WMMA_N;
    constexpr int BLOCK_K = 32;

    // 双缓冲共享内存
    __shared__ half As[2][BLOCK_M][BLOCK_K];
    __shared__ half Bs[2][BLOCK_K][BLOCK_N];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_COL_WARPS;
    int warpCol = warpId % BLOCK_COL_WARPS;

    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;

    // 累加器
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> 
        acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    // 预加载第一块到缓冲区 0
    int bufIdx = 0;
    for (int loadIdx = threadIdx.x; loadIdx < BLOCK_M * BLOCK_K; loadIdx += blockDim.x) {
        int row = loadIdx / BLOCK_K;
        int col = loadIdx % BLOCK_K;
        int globalRow = blockRowStart + row;
        int globalCol = col;
        As[bufIdx][row][col] = (globalRow < M && globalCol < K) ? 
                               A[globalRow * K + globalCol] : __float2half(0.0f);
    }
    for (int loadIdx = threadIdx.x; loadIdx < BLOCK_K * BLOCK_N; loadIdx += blockDim.x) {
        int row = loadIdx / BLOCK_N;
        int col = loadIdx % BLOCK_N;
        int globalRow = row;
        int globalCol = blockColStart + col;
        Bs[bufIdx][row][col] = (globalRow < K && globalCol < N) ? 
                               B[globalRow * N + globalCol] : __float2half(0.0f);
    }

    __syncthreads();

    // 主循环：计算当前块，同时加载下一块
    for (int k = 0; k < K; k += BLOCK_K) {
        int nextK = k + BLOCK_K;
        int nextBufIdx = 1 - bufIdx;

        // 异步加载下一块（如果存在）
        if (nextK < K) {
            for (int loadIdx = threadIdx.x; loadIdx < BLOCK_M * BLOCK_K; loadIdx += blockDim.x) {
                int row = loadIdx / BLOCK_K;
                int col = loadIdx % BLOCK_K;
                int globalRow = blockRowStart + row;
                int globalCol = nextK + col;
                As[nextBufIdx][row][col] = (globalRow < M && globalCol < K) ? 
                                           A[globalRow * K + globalCol] : __float2half(0.0f);
            }
            for (int loadIdx = threadIdx.x; loadIdx < BLOCK_K * BLOCK_N; loadIdx += blockDim.x) {
                int row = loadIdx / BLOCK_N;
                int col = loadIdx % BLOCK_N;
                int globalRow = nextK + row;
                int globalCol = blockColStart + col;
                Bs[nextBufIdx][row][col] = (globalRow < K && globalCol < N) ? 
                                           B[globalRow * N + globalCol] : __float2half(0.0f);
            }
        }

        // 计算当前块
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_ROW_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_COL_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                int row = warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[bufIdx][row][kk], BLOCK_K);
            }

            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++) {
                int col = warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[bufIdx][kk][col], BLOCK_N);
            }

            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++) {
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
                }
            }
        }

        __syncthreads();
        bufIdx = nextBufIdx;
    }

    // 存储结果
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            int cRow = blockRowStart + warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
            int cCol = blockColStart + warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
            if (cRow < M && cCol < N) {
                wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// ============================================================================
// 版本4: 向量化加载 + 寄存器优化
// 使用向量化内存访问提高带宽利用率
// ============================================================================
template<int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, int WARP_ROW_TILES, int WARP_COL_TILES>
__global__ void wmma_gemm_v4_vectorized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int BLOCK_ROW_TILES = BLOCK_ROW_WARPS * WARP_ROW_TILES;
    constexpr int BLOCK_COL_TILES = BLOCK_COL_WARPS * WARP_COL_TILES;
    constexpr int BLOCK_M = BLOCK_ROW_TILES * WMMA_M;
    constexpr int BLOCK_N = BLOCK_COL_TILES * WMMA_N;
    constexpr int BLOCK_K = 32;
    constexpr int CHUNK_K = 2;  // 每次处理 2 个 WMMA_K

    // 使用 padding 避免 bank conflict
    __shared__ half As[BLOCK_M][BLOCK_K + 8];
    __shared__ half Bs[BLOCK_K][BLOCK_N + 8];

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = warpId / BLOCK_COL_WARPS;
    int warpCol = warpId % BLOCK_COL_WARPS;

    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;

    // 累加器
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> 
        acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    // 主循环
    for (int k = 0; k < K; k += BLOCK_K) {
        // 向量化加载 A (使用 float4 = 8 个 half)
        for (int loadIdx = threadIdx.x; loadIdx < (BLOCK_M * BLOCK_K) / 8; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_K;
            int col = (loadIdx * 8) % BLOCK_K;
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
            
            if (globalRow < M && globalCol + 7 < K) {
                // 向量化加载 8 个 half
                float4 tmp = *reinterpret_cast<const float4*>(&A[globalRow * K + globalCol]);
                *reinterpret_cast<float4*>(&As[row][col]) = tmp;
            } else {
                // 边界处理
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    As[row][col + i] = (globalRow < M && gc < K) ? 
                                       A[globalRow * K + gc] : __float2half(0.0f);
                }
            }
        }

        // 向量化加载 B
        for (int loadIdx = threadIdx.x; loadIdx < (BLOCK_K * BLOCK_N) / 8; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_N;
            int col = (loadIdx * 8) % BLOCK_N;
            int globalRow = k + row;
            int globalCol = blockColStart + col;
            
            if (globalRow < K && globalCol + 7 < N) {
                float4 tmp = *reinterpret_cast<const float4*>(&B[globalRow * N + globalCol]);
                *reinterpret_cast<float4*>(&Bs[row][col]) = tmp;
            } else {
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    Bs[row][col + i] = (globalRow < K && gc < N) ? 
                                       B[globalRow * N + gc] : __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_ROW_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_COL_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                int row = warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[row][kk], BLOCK_K + 8);
            }

            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++) {
                int col = warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][col], BLOCK_N + 8);
            }

            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++) {
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // 向量化存储结果
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            int cRow = blockRowStart + warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
            int cCol = blockColStart + warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
            if (cRow < M && cCol < N) {
                wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// Wrapper 函数
void launch_wmma_gemm_v1(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(32, 4);  // 4 warps per block
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M / 4);
    wmma_gemm_v1_naive<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_wmma_gemm_v2(const half* A, const half* B, float* C, int M, int N, int K) {
    // block process 2 warps in row direction
    constexpr int BLOCK_ROW_WARPS = 2;
    constexpr int BLOCK_COL_WARPS = 2;
    // warp process 2 tiles in row direction
    constexpr int WARP_ROW_TILES = 2;
    constexpr int WARP_COL_TILES = 2;
    constexpr int BLOCK_M = BLOCK_ROW_WARPS * WARP_ROW_TILES * WMMA_M;  // 64
    constexpr int BLOCK_N = BLOCK_COL_WARPS * WARP_COL_TILES * WMMA_N;  // 64
    
    dim3 block(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);  // 128 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    wmma_gemm_v2_shared<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
        <<<grid, block>>>(A, B, C, M, N, K);
}

void launch_wmma_gemm_v3(const half* A, const half* B, float* C, int M, int N, int K) {
    constexpr int BLOCK_ROW_WARPS = 2;
    constexpr int BLOCK_COL_WARPS = 2;
    constexpr int WARP_ROW_TILES = 2;
    constexpr int WARP_COL_TILES = 2;
    constexpr int BLOCK_M = BLOCK_ROW_WARPS * WARP_ROW_TILES * WMMA_M;
    constexpr int BLOCK_N = BLOCK_COL_WARPS * WARP_COL_TILES * WMMA_N;
    
    dim3 block(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    wmma_gemm_v3_double_buffer<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
        <<<grid, block>>>(A, B, C, M, N, K);
}

void launch_wmma_gemm_v4(const half* A, const half* B, float* C, int M, int N, int K) {
    constexpr int BLOCK_ROW_WARPS = 2;
    constexpr int BLOCK_COL_WARPS = 2;
    constexpr int WARP_ROW_TILES = 2;
    constexpr int WARP_COL_TILES = 2;
    constexpr int BLOCK_M = BLOCK_ROW_WARPS * WARP_ROW_TILES * WMMA_M;
    constexpr int BLOCK_N = BLOCK_COL_WARPS * WARP_COL_TILES * WMMA_N;
    
    dim3 block(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    wmma_gemm_v4_vectorized<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
        <<<grid, block>>>(A, B, C, M, N, K);
}

