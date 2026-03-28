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
//
// English: One block covers four 16x16 output tiles stacked vertically along
// the same column of C; blockIdx.x selects the tile column, blockIdx.y selects
// the next group of four tile-rows (along M).
// ============================================================================
__global__ void wmma_gemm_v1_naive(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // One block = four 16x16 tiles in one column of C (stacked along M);
    // blockIdx.x -> tile column; blockIdx.y -> which group of four tile-rows.
    // -------------------------------------------------------------------------
    // 输出 C 按 16×16 切块，(warpM, warpN) = 当前 warp 负责的那一块在「块网格」里的索引。
    // block(32,4)：threadIdx.x∈[0,31] 为一个 warp；threadIdx.y∈[0,3] 共 4 个 warp。
    // 同一 block 内 4 个 warp 的 warpN 相同（同一列块），warpM 相差 0,1,2,3（上下叠 4 行块）。
    // -------------------------------------------------------------------------
    // warpM：M 方向第几块。blockIdx.y 是「竖条组」编号，每组占 4 行块；threadIdx.y 是组内第几行。
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;
    // warpN：N 方向第几块。本布局下每 block 只在 N 上占一列块，故 warpN == blockIdx.x
    // （原式 (blockIdx.x*32+threadIdx.x)/32 在 threadIdx.x<32 时恒等于 blockIdx.x，32 来自 warp 宽度）。
    int warpN = blockIdx.x;
 
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

    int warpId = threadIdx.x / 32;
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
    __shared__ half As[BLOCK_M][BLOCK_K + 8];
    __shared__ half Bs[BLOCK_K][BLOCK_N + 8];

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

// ============================================================================
// 版本5: 增大 Block Tile 到 128×128
// 8 warps (4×2), 每 warp 处理 4×2 tiles → 128×128 block
// 计算/访存比从 64 提升到 128 FLOPs/byte
// ============================================================================
template<int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, int WARP_ROW_TILES, int WARP_COL_TILES>
__global__ void wmma_gemm_v5_large_tile(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int BLOCK_ROW_TILES = BLOCK_ROW_WARPS * WARP_ROW_TILES;
    constexpr int BLOCK_COL_TILES = BLOCK_COL_WARPS * WARP_COL_TILES;
    constexpr int BLOCK_M = BLOCK_ROW_TILES * WMMA_M;   // 4*4*16 = 256 → actually depends on template params
    constexpr int BLOCK_N = BLOCK_COL_TILES * WMMA_N;
    constexpr int BLOCK_K = 32;

    __shared__ half As[BLOCK_M][BLOCK_K + 8];
    __shared__ half Bs[BLOCK_K][BLOCK_N + 8];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_COL_WARPS;
    int warpCol = warpId % BLOCK_COL_WARPS;

    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++)
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++)
            wmma::fill_fragment(acc_frag[i][j], 0.0f);

    for (int k = 0; k < K; k += BLOCK_K) {
        // 向量化加载 A
        for (int loadIdx = threadIdx.x; loadIdx < (BLOCK_M * BLOCK_K) / 8; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_K;
            int col = (loadIdx * 8) % BLOCK_K;
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
            if (globalRow < M && globalCol + 7 < K) {
                *reinterpret_cast<float4*>(&As[row][col]) =
                    *reinterpret_cast<const float4*>(&A[globalRow * K + globalCol]);
            } else {
                #pragma unroll
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
                *reinterpret_cast<float4*>(&Bs[row][col]) =
                    *reinterpret_cast<const float4*>(&B[globalRow * N + globalCol]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    Bs[row][col + i] = (globalRow < K && gc < N) ?
                        B[globalRow * N + gc] : __float2half(0.0f);
                }
            }
        }

        __syncthreads();

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
            for (int i = 0; i < WARP_ROW_TILES; i++)
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++)
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            int cRow = blockRowStart + warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
            int cCol = blockColStart + warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
            if (cRow < M && cCol < N)
                wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// ============================================================================
// 版本6: cp.async 异步拷贝 + 多阶段流水线
// 使用硬件异步拷贝 Global→Shared，配合 3 阶段流水线深度隐藏延迟
// 需要 Compute 8.0+ (Ampere)
// ============================================================================
template<int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, int WARP_ROW_TILES, int WARP_COL_TILES, int NUM_STAGES>
__global__ void wmma_gemm_v6_async_pipeline(
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

    // 多阶段共享内存缓冲
    __shared__ half As[NUM_STAGES][BLOCK_M][BLOCK_K + 8];
    __shared__ half Bs[NUM_STAGES][BLOCK_K][BLOCK_N + 8];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_COL_WARPS;
    int warpCol = warpId % BLOCK_COL_WARPS;

    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++)
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++)
            wmma::fill_fragment(acc_frag[i][j], 0.0f);

    // cp.async helper: 从 global 直接拷贝 16 字节到 shared，不经过寄存器
    auto cp_async_16B = [](void* smem_ptr, const void* gmem_ptr) {
        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_ptr));
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"(gmem_ptr));
    };

    auto async_load_stage = [&](int stage, int k_offset) {
        // 每次加载 16 字节 = 8 个 half
        constexpr int A_TOTAL = (BLOCK_M * BLOCK_K) / 8;
        constexpr int B_TOTAL = (BLOCK_K * BLOCK_N) / 8;

        for (int loadIdx = threadIdx.x; loadIdx < A_TOTAL; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_K;
            int col = (loadIdx * 8) % BLOCK_K;
            int globalRow = blockRowStart + row;
            int globalCol = k_offset + col;
            if (globalRow < M && globalCol + 7 < K) {
                cp_async_16B(&As[stage][row][col], &A[globalRow * K + globalCol]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    As[stage][row][col + i] = (globalRow < M && gc < K) ?
                        A[globalRow * K + gc] : __float2half(0.0f);
                }
            }
        }

        for (int loadIdx = threadIdx.x; loadIdx < B_TOTAL; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_N;
            int col = (loadIdx * 8) % BLOCK_N;
            int globalRow = k_offset + row;
            int globalCol = blockColStart + col;
            if (globalRow < K && globalCol + 7 < N) {
                cp_async_16B(&Bs[stage][row][col], &B[globalRow * N + globalCol]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    Bs[stage][row][col + i] = (globalRow < K && gc < N) ?
                        B[globalRow * N + gc] : __float2half(0.0f);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    auto compute_stage = [&](int stage) {
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_ROW_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_COL_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                int row = warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[stage][row][kk], BLOCK_K + 8);
            }
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++) {
                int col = warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[stage][kk][col], BLOCK_N + 8);
            }
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++)
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++)
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
        }
    };

    int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // 预填充流水线：发射前 NUM_STAGES 个异步加载
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        if (s < num_k_blocks) {
            async_load_stage(s, s * BLOCK_K);
        }
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int stage = k_block % NUM_STAGES;

        // 排空阶段（不再发 prefetch）需要更严格的 wait 以确保数据就绪
        if (k_block + (NUM_STAGES - 1) >= num_k_blocks) {
            asm volatile("cp.async.wait_group 0;\n");
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));
        }
        __syncthreads();

        compute_stage(stage);
        __syncthreads();

        int prefetch_block = k_block + NUM_STAGES;
        if (prefetch_block < num_k_blocks) {
            async_load_stage(stage, prefetch_block * BLOCK_K);
        }
    }

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            int cRow = blockRowStart + warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
            int cCol = blockColStart + warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
            if (cRow < M && cCol < N)
                wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// ============================================================================
// 版本7: 终极组合 — 大 Tile + cp.async + 向量化 + Padding
// 把 V5 (大 tile 128×64) 和 V6 (cp.async 3阶段流水线) 结合
// ============================================================================
template<int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, int WARP_ROW_TILES, int WARP_COL_TILES, int NUM_STAGES>
__global__ void wmma_gemm_v7_combined(
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

    __shared__ half As[NUM_STAGES][BLOCK_M][BLOCK_K + 8];
    __shared__ half Bs[NUM_STAGES][BLOCK_K][BLOCK_N + 8];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_COL_WARPS;
    int warpCol = warpId % BLOCK_COL_WARPS;

    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++)
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++)
            wmma::fill_fragment(acc_frag[i][j], 0.0f);

    auto cp_async_16B = [](void* smem_ptr, const void* gmem_ptr) {
        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_ptr));
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"(gmem_ptr));
    };

    auto async_load_stage = [&](int stage, int k_offset) {
        constexpr int A_TOTAL = (BLOCK_M * BLOCK_K) / 8;
        constexpr int B_TOTAL = (BLOCK_K * BLOCK_N) / 8;

        for (int loadIdx = threadIdx.x; loadIdx < A_TOTAL; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_K;
            int col = (loadIdx * 8) % BLOCK_K;
            int globalRow = blockRowStart + row;
            int globalCol = k_offset + col;
            if (globalRow < M && globalCol + 7 < K) {
                cp_async_16B(&As[stage][row][col], &A[globalRow * K + globalCol]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    As[stage][row][col + i] = (globalRow < M && gc < K) ?
                        A[globalRow * K + gc] : __float2half(0.0f);
                }
            }
        }

        for (int loadIdx = threadIdx.x; loadIdx < B_TOTAL; loadIdx += blockDim.x) {
            int row = (loadIdx * 8) / BLOCK_N;
            int col = (loadIdx * 8) % BLOCK_N;
            int globalRow = k_offset + row;
            int globalCol = blockColStart + col;
            if (globalRow < K && globalCol + 7 < N) {
                cp_async_16B(&Bs[stage][row][col], &B[globalRow * N + globalCol]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gc = globalCol + i;
                    Bs[stage][row][col + i] = (globalRow < K && gc < N) ?
                        B[globalRow * N + gc] : __float2half(0.0f);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        if (s < num_k_blocks)
            async_load_stage(s, s * BLOCK_K);
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int stage = k_block % NUM_STAGES;

        if (k_block + (NUM_STAGES - 1) >= num_k_blocks) {
            asm volatile("cp.async.wait_group 0;\n");
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_ROW_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_COL_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++) {
                int row = warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[stage][row][kk], BLOCK_K + 8);
            }
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++) {
                int col = warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[stage][kk][col], BLOCK_N + 8);
            }
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++)
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++)
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
        }

        __syncthreads();

        int prefetch_block = k_block + NUM_STAGES;
        if (prefetch_block < num_k_blocks)
            async_load_stage(stage, prefetch_block * BLOCK_K);
    }

    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
            int cRow = blockRowStart + warpRow * WARP_ROW_TILES * WMMA_M + i * WMMA_M;
            int cCol = blockColStart + warpCol * WARP_COL_TILES * WMMA_N + j * WMMA_N;
            if (cRow < M && cCol < N)
                wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// Wrapper 函数
// Grid/block: one block = four 16x16 tiles stacked in the same column of C;
// blockIdx.x steps along N (tile columns), blockIdx.y steps along groups of four tile-rows (M).
void launch_wmma_gemm_v1(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(32, 4);  // 4 warps per block；x=32 为 warp 宽度，y=4 为每 block 在 M 方向占 4 个 16×16 块
    // grid.x：N 方向需要多少「列」块，与 warpN = blockIdx.x 一致。
    // grid.y：M 方向需要多少「组」，每组 4 行块（对应 4 个 warp），须 ceil(tileRows/4)，不能整除向下取整。
    const int tile_cols = (N + WMMA_N - 1) / WMMA_N;
    const int tile_rows = (M + WMMA_M - 1) / WMMA_M;
    const int grid_y = (tile_rows + 4 - 1) / 4;
    dim3 grid(tile_cols, grid_y);
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

// V5: 大 tile 128×64, 8 warps (4×2), 每 warp 2×2 tiles
void launch_wmma_gemm_v5(const half* A, const half* B, float* C, int M, int N, int K) {
    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 2;
    constexpr int WARP_ROW_TILES = 2;
    constexpr int WARP_COL_TILES = 2;
    constexpr int BLOCK_M = BLOCK_ROW_WARPS * WARP_ROW_TILES * WMMA_M;  // 128
    constexpr int BLOCK_N = BLOCK_COL_WARPS * WARP_COL_TILES * WMMA_N;  // 64
    
    dim3 block(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);  // 256 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    wmma_gemm_v5_large_tile<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
        <<<grid, block>>>(A, B, C, M, N, K);
}

// V6: cp.async 3阶段流水线, 64×64 tile
void launch_wmma_gemm_v6(const half* A, const half* B, float* C, int M, int N, int K) {
    constexpr int BLOCK_ROW_WARPS = 2;
    constexpr int BLOCK_COL_WARPS = 2;
    constexpr int WARP_ROW_TILES = 2;
    constexpr int WARP_COL_TILES = 2;
    constexpr int NUM_STAGES = 3;
    constexpr int BLOCK_M = BLOCK_ROW_WARPS * WARP_ROW_TILES * WMMA_M;
    constexpr int BLOCK_N = BLOCK_COL_WARPS * WARP_COL_TILES * WMMA_N;

    dim3 block(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);  // 128 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    // V6 共享内存 = 3 stages × (64×40 + 32×72) × 2 bytes ≈ 28.5 KB
    // 需要设置较大的共享内存
    cudaFuncSetAttribute(
        wmma_gemm_v6_async_pipeline<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, NUM_STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

    wmma_gemm_v6_async_pipeline<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, NUM_STAGES>
        <<<grid, block>>>(A, B, C, M, N, K);
}

// V7: 终极组合 — 大 tile 128×64 + cp.async 3阶段
void launch_wmma_gemm_v7(const half* A, const half* B, float* C, int M, int N, int K) {
    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 2;
    constexpr int WARP_ROW_TILES = 2;
    constexpr int WARP_COL_TILES = 2;
    constexpr int NUM_STAGES = 3;
    constexpr int BLOCK_M = BLOCK_ROW_WARPS * WARP_ROW_TILES * WMMA_M;  // 128
    constexpr int BLOCK_N = BLOCK_COL_WARPS * WARP_COL_TILES * WMMA_N;  // 64

    dim3 block(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);  // 256 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    wmma_gemm_v7_combined<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, NUM_STAGES>
        <<<grid, block>>>(A, B, C, M, N, K);
}

