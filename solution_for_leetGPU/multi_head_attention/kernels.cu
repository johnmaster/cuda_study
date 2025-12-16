#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(42);
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

// Warp-level reduction for finding maximum
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Compute Q_h @ K_h^T for all heads and batches, then scale by 1/sqrt(d_h)
 * 
 * Input layout: Q, K are [B, N, d] in row-major order
 * Each head processes Q[:, :, h*d_h:(h+1)*d_h] and K[:, :, h*d_h:(h+1)*d_h]
 * 
 * Output: S[b, h, i, j] = sum_k(Q[b, i, h*d_h + k] * K[b, j, h*d_h + k]) / sqrt(d_h)
 * Output layout: [B, num_heads, N, N]
 */
__global__ void computeQKTransposeKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int B, int N, int d, int num_heads
) {
    int d_h = d / num_heads;
    float scale = 1.0f / sqrtf((float)d_h);
    
    // Each block handles one (batch, head) pair
    int bh = blockIdx.z;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    
    // Each thread computes one element S[batch, head, row, col]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < B && row < N && col < N) {
        float sum = 0.0f;
        
        // Compute dot product over head dimension
        int q_base = batch * N * d + row * d + head * d_h;
        int k_base = batch * N * d + col * d + head * d_h;
        
        #pragma unroll 4
        for (int k = 0; k < d_h; k++) {
            sum += Q[q_base + k] * K[k_base + k];
        }
        
        // Scale by 1/sqrt(d_h)
        sum *= scale;
        
        // Store in S[batch, head, row, col]
        int s_idx = batch * num_heads * N * N + head * N * N + row * N + col;
        S[s_idx] = sum;
    }
}

/**
 * Tiled version of Q @ K^T computation with shared memory
 */
template<int TILE>
__global__ void computeQKTransposeTiledKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int B, int N, int d, int num_heads
) {
    __shared__ float tile_Q[TILE][TILE + 1];  // +1 to avoid bank conflicts
    __shared__ float tile_K[TILE][TILE + 1];
    
    int d_h = d / num_heads;
    float scale = 1.0f / sqrtf((float)d_h);
    
    // Each block handles one (batch, head) pair
    int bh = blockIdx.z;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    int num_tiles = (d_h + TILE - 1) / TILE;
    
    for (int t = 0; t < num_tiles; t++) {
        int k_idx = t * TILE + threadIdx.x;
        int k_idy = t * TILE + threadIdx.y;
        
        // Load Q tile: Q[batch, row, head*d_h + k]
        if (row < N && k_idx < d_h) {
            tile_Q[threadIdx.y][threadIdx.x] = Q[batch * N * d + row * d + head * d_h + k_idx];
        } else {
            tile_Q[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load K tile (transposed access): K[batch, col, head*d_h + k]
        if (col < N && k_idy < d_h) {
            tile_K[threadIdx.y][threadIdx.x] = K[batch * N * d + col * d + head * d_h + k_idy];
        } else {
            tile_K[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += tile_Q[threadIdx.y][k] * tile_K[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Scale and store
    if (batch < B && row < N && col < N) {
        int s_idx = batch * num_heads * N * N + head * N * N + row * N + col;
        S[s_idx] = sum * scale;
    }
}

/**
 * Row-wise softmax kernel
 * Input: S[B, num_heads, N, N]
 * Output: A[B, num_heads, N, N] where each row is softmaxed
 * 
 * Each block handles one row (B * num_heads * N total rows)
 */
__global__ void softmaxRowKernel(
    const float* __restrict__ S,
    float* __restrict__ A,
    int total_rows,  // B * num_heads * N
    int N            // row length
) {
    extern __shared__ float sdata[];
    
    int row_idx = blockIdx.x;
    if (row_idx >= total_rows) return;
    
    const float* row_in = S + row_idx * N;
    float* row_out = A + row_idx * N;
    
    // Step 1: Find row maximum
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    
    // Warp reduction for max
    max_val = warpReduceMax(max_val);
    
    // Block reduction for max
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();
    
    // First warp reduces the partial maxes
    if (wid == 0) {
        max_val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : -INFINITY;
        max_val = warpReduceMax(max_val);
    }
    
    // Broadcast max to all threads
    if (threadIdx.x == 0) {
        sdata[0] = max_val;
    }
    __syncthreads();
    max_val = sdata[0];
    
    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float exp_val = expf(row_in[i] - max_val);
        row_out[i] = exp_val;  // Store intermediate
        sum += exp_val;
    }
    
    // Warp reduction for sum
    sum = warpReduceSum(sum);
    
    // Block reduction for sum
    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();
    
    if (wid == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    
    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        sdata[0] = sum;
    }
    __syncthreads();
    sum = sdata[0];
    
    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

/**
 * Compute A @ V for all heads and batches
 * 
 * A: [B, num_heads, N, N] - attention weights
 * V: [B, N, d] - value matrix (need to extract per-head slice)
 * 
 * Output: O[b, h, i, k] = sum_j(A[b, h, i, j] * V[b, j, h*d_h + k])
 * Stored as: output[b, i, h*d_h + k] (final [B, N, d] layout)
 */
__global__ void computeAVKernel(
    const float* __restrict__ A,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int N, int d, int num_heads
) {
    int d_h = d / num_heads;
    
    // Each block handles one (batch, head) pair
    int bh = blockIdx.z;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    
    // Each thread computes one element output[batch, row, head*d_h + k]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // within d_h
    
    if (batch < B && row < N && k < d_h) {
        float sum = 0.0f;
        
        // A[batch, head, row, j] * V[batch, j, head*d_h + k]
        int a_base = batch * num_heads * N * N + head * N * N + row * N;
        
        #pragma unroll 4
        for (int j = 0; j < N; j++) {
            float a_val = A[a_base + j];
            float v_val = V[batch * N * d + j * d + head * d_h + k];
            sum += a_val * v_val;
        }
        
        // Store in output[batch, row, head*d_h + k]
        output[batch * N * d + row * d + head * d_h + k] = sum;
    }
}

/**
 * Tiled version of A @ V computation
 */
template<int TILE>
__global__ void computeAVTiledKernel(
    const float* __restrict__ A,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int N, int d, int num_heads
) {
    __shared__ float tile_A[TILE][TILE + 1];
    __shared__ float tile_V[TILE][TILE + 1];
    
    int d_h = d / num_heads;
    
    int bh = blockIdx.z;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int k = blockIdx.x * TILE + threadIdx.x;  // within d_h
    
    float sum = 0.0f;
    
    int num_tiles = (N + TILE - 1) / TILE;
    
    for (int t = 0; t < num_tiles; t++) {
        int j = t * TILE + threadIdx.x;
        int j2 = t * TILE + threadIdx.y;
        
        // Load A tile: A[batch, head, row, j]
        if (row < N && j < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[batch * num_heads * N * N + head * N * N + row * N + j];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load V tile: V[batch, j, head*d_h + k]
        if (j2 < N && k < d_h) {
            tile_V[threadIdx.y][threadIdx.x] = V[batch * N * d + j2 * d + head * d_h + k];
        } else {
            tile_V[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += tile_A[threadIdx.y][i] * tile_V[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (batch < B && row < N && k < d_h) {
        output[batch * N * d + row * d + head * d_h + k] = sum;
    }
}

void solve(const float* Q, const float* K, const float* V, float* output,
           int B, int N, int d, int num_heads) {
    
    if (B == 0 || N == 0 || d == 0 || num_heads == 0) return;
    
    int d_h = d / num_heads;
    
    // Allocate intermediate storage for attention scores S and attention weights A
    // S, A: [B, num_heads, N, N]
    size_t attn_size = (size_t)B * num_heads * N * N;
    float* d_S;
    float* d_A;
    CHECK_CUDA(cudaMalloc(&d_S, attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A, attn_size * sizeof(float)));
    
    // Step 1: Compute Q @ K^T / sqrt(d_h) for all heads
    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, B * num_heads);
        
        if (d_h <= 64) {
            // For small head dimensions, use simple kernel
            computeQKTransposeKernel<<<grid, block>>>(Q, K, d_S, B, N, d, num_heads);
        } else {
            // For larger head dimensions, use tiled kernel
            computeQKTransposeTiledKernel<TILE><<<grid, block>>>(Q, K, d_S, B, N, d, num_heads);
        }
    }
    
    // Step 2: Apply row-wise softmax to get attention weights
    {
        int total_rows = B * num_heads * N;
        int block_size = min(BLOCK_SIZE, ((N + 31) / 32) * 32);
        block_size = max(block_size, 32);
        int shared_mem = (block_size / 32 + 1) * sizeof(float);
        
        softmaxRowKernel<<<total_rows, block_size, shared_mem>>>(d_S, d_A, total_rows, N);
    }
    
    // Step 3: Compute A @ V and write to output
    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((d_h + TILE - 1) / TILE, (N + TILE - 1) / TILE, B * num_heads);
        
        if (N <= 64) {
            computeAVKernel<<<grid, block>>>(d_A, V, output, B, N, d, num_heads);
        } else {
            computeAVTiledKernel<TILE><<<grid, block>>>(d_A, V, output, B, N, d, num_heads);
        }
    }
    
    // Free intermediate storage
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_A));
}

