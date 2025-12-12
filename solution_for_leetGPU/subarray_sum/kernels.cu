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
 * Subarray Sum Kernel
 * 
 * Computes the sum of elements in input[S..E] (inclusive).
 * 
 * Strategy:
 * 1. Each thread loads elements within the range [S, E]
 * 2. Use warp shuffle for fast intra-warp reduction
 * 3. Store warp results to shared memory
 * 4. Final reduction of warp results
 * 5. Atomic add to accumulate block results
 */
__global__ void subarray_sum_kernel(int32_t* d_input, int64_t* d_output, size_t S, size_t E) {
    __shared__ int64_t warp_sums[32];  // Max 32 warps per block (1024 threads / 32)
    
    size_t tid = threadIdx.x;
    size_t range_len = E - S + 1;
    
    // Each block processes blockDim.x * 2 elements for better efficiency
    size_t local_idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Each thread accumulates up to 2 elements
    int64_t sum = 0;
    
    if (local_idx < range_len) {
        sum = d_input[S + local_idx];
    }
    if (local_idx + blockDim.x < range_len) {
        sum += d_input[S + local_idx + blockDim.x];
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
 * Computes sum of subarray: result = sum(input[i]) for i = S to E (inclusive)
 * 
 * Uses int64_t internally to prevent overflow when summing many int32_t values
 */
void solve(int32_t* d_input, int32_t* d_output, size_t n, size_t S, size_t E) {
    // Handle edge cases
    if (S > E || S >= n) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int32_t)));
        return;
    }
    
    // Clamp E to valid range
    if (E >= n) {
        E = n - 1;
    }
    
    size_t range_len = E - S + 1;
    
    if (range_len == 0) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int32_t)));
        return;
    }
    
    // Allocate temporary int64_t for accumulation to prevent overflow
    int64_t* d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, sizeof(int64_t)));
    CHECK_CUDA(cudaMemset(d_temp, 0, sizeof(int64_t)));
    
    // Each thread processes 2 elements
    size_t elements_per_block = BLOCK_SIZE * 2;
    int num_blocks = (range_len + elements_per_block - 1) / elements_per_block;
    
    subarray_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_temp, S, E);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back to int32_t output (truncating if overflow)
    int64_t h_temp;
    CHECK_CUDA(cudaMemcpy(&h_temp, d_temp, sizeof(int64_t), cudaMemcpyDeviceToHost));
    int32_t result = static_cast<int32_t>(h_temp);
    CHECK_CUDA(cudaMemcpy(d_output, &result, sizeof(int32_t), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaFree(d_temp));
}
