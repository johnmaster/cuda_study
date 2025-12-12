#include "kernels.cuh"

void rand_init(int32_t* h_data, size_t n) {
    std::mt19937 gen(200);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);
    for (size_t i = 0; i < n; i++) {
        h_data[i] = dist(gen);
    }
}

int32_t* alloc_host(size_t n) {
    return new int32_t[n];
}

int32_t* alloc_device(size_t n) {
    int32_t* p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(int32_t)));
    return p;
}

void free_host(int32_t* p) {
    delete[] p;
}

void free_device(int32_t* p) {
    CHECK_CUDA(cudaFree(p));
}

/*
 * Warp-level reduction using shuffle instructions
 * Each thread in a warp exchanges values with other threads
 * and accumulates them to produce a partial sum
 */
__device__ int64_t warp_reduce_sum(int64_t val) {
    // Use shuffle down to reduce within a warp (32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * 2D Subarray Sum Kernel
 * 
 * Computes the sum of elements in input[S_ROW..E_ROW][S_COL..E_COL] (inclusive).
 * The input is stored in row-major order with M columns per row.
 * 
 * Strategy:
 * 1. Linearize the 2D subarray into a 1D iteration space
 * 2. Each thread processes elements based on global linear index
 * 3. Convert linear index back to 2D coordinates to access correct elements
 * 4. Use warp shuffle for fast intra-warp reduction
 * 5. Store warp results to shared memory
 * 6. Final reduction of warp results
 * 7. Atomic add to accumulate block results
 */
__global__ void subarray_sum_2d_kernel(int32_t* d_input, int64_t* d_output,
                                        size_t M,
                                        size_t S_ROW, size_t E_ROW,
                                        size_t S_COL, size_t E_COL,
                                        size_t total_elements) {
    __shared__ int64_t warp_sums[32];  // Max 32 warps per block (1024 threads / 32)
    
    size_t tid = threadIdx.x;
    
    // Number of columns in the subarray
    size_t sub_cols = E_COL - S_COL + 1;
    
    // Each block processes blockDim.x * 2 elements for better efficiency
    size_t local_idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Each thread accumulates up to 2 elements
    int64_t sum = 0;
    
    // Process first element
    if (local_idx < total_elements) {
        // Convert linear index to 2D coordinates within the subarray
        size_t sub_row = local_idx / sub_cols;
        size_t sub_col = local_idx % sub_cols;
        
        // Map to actual array coordinates
        size_t actual_row = S_ROW + sub_row;
        size_t actual_col = S_COL + sub_col;
        
        // Linear index in the full array (row-major)
        size_t array_idx = actual_row * M + actual_col;
        sum = d_input[array_idx];
    }
    
    // Process second element
    if (local_idx + blockDim.x < total_elements) {
        size_t idx2 = local_idx + blockDim.x;
        
        // Convert linear index to 2D coordinates within the subarray
        size_t sub_row = idx2 / sub_cols;
        size_t sub_col = idx2 % sub_cols;
        
        // Map to actual array coordinates
        size_t actual_row = S_ROW + sub_row;
        size_t actual_col = S_COL + sub_col;
        
        // Linear index in the full array (row-major)
        size_t array_idx = actual_row * M + actual_col;
        sum += d_input[array_idx];
    }
    
    // Warp-level reduction using shuffle
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warp_reduce_sum(sum);
    
    // First thread in each warp stores the warp's sum
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction: first warp reduces all warp sums
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warp_sums[tid] : 0;
        sum = warp_reduce_sum(sum);
    }
    
    // Thread 0 atomically adds this block's result to the output
    if (tid == 0) {
        atomicAdd((unsigned long long*)d_output, (unsigned long long)sum);
    }
}

/*
 * Main solve function
 * Computes sum of 2D subarray: result = sum(input[r][c]) 
 * for r in [S_ROW, E_ROW] and c in [S_COL, E_COL] (inclusive)
 * 
 * Uses int64_t internally to prevent overflow when summing many int32_t values
 */
void solve(int32_t* d_input, int32_t* d_output, size_t N, size_t M,
           size_t S_ROW, size_t E_ROW, size_t S_COL, size_t E_COL) {
    // Handle edge cases
    if (S_ROW > E_ROW || S_COL > E_COL || S_ROW >= N || S_COL >= M) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int32_t)));
        return;
    }
    
    // Clamp indices to valid ranges
    if (E_ROW >= N) {
        E_ROW = N - 1;
    }
    if (E_COL >= M) {
        E_COL = M - 1;
    }
    
    // Calculate total number of elements in the 2D subarray
    size_t num_rows = E_ROW - S_ROW + 1;
    size_t num_cols = E_COL - S_COL + 1;
    size_t total_elements = num_rows * num_cols;
    
    if (total_elements == 0) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int32_t)));
        return;
    }
    
    // Allocate temporary int64_t for accumulation to prevent overflow
    int64_t* d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, sizeof(int64_t)));
    CHECK_CUDA(cudaMemset(d_temp, 0, sizeof(int64_t)));
    
    // Each thread processes 2 elements
    size_t elements_per_block = BLOCK_SIZE * 2;
    int num_blocks = (total_elements + elements_per_block - 1) / elements_per_block;
    
    subarray_sum_2d_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_temp, M, S_ROW, E_ROW, S_COL, E_COL, total_elements
    );
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back to int32_t output (truncating if overflow)
    int64_t h_temp;
    CHECK_CUDA(cudaMemcpy(&h_temp, d_temp, sizeof(int64_t), cudaMemcpyDeviceToHost));
    int32_t result = static_cast<int32_t>(h_temp);
    CHECK_CUDA(cudaMemcpy(d_output, &result, sizeof(int32_t), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaFree(d_temp));
}
