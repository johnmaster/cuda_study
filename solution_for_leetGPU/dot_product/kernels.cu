#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(200);
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

/*
 * Warp-level reduction using shuffle instructions
 * Each thread in a warp exchanges values with other threads
 * and accumulates them to produce a partial sum
 */
__device__ float warp_reduce_sum(float val) {
    // Use shuffle down to reduce within a warp (32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * Dot Product Kernel
 * 
 * Each thread computes the product of corresponding elements from vectors a and b,
 * then all products are reduced (summed) to produce the final dot product.
 * 
 * Strategy:
 * 1. Each thread loads and multiplies elements from both vectors
 * 2. Use warp shuffle for fast intra-warp reduction
 * 3. Store warp results to shared memory
 * 4. Final reduction of warp results
 * 5. Atomic add to accumulate block results
 */
__global__ void dot_product_kernel(float* d_output, float* d_a, float* d_b, size_t n) {
    __shared__ float warp_sums[32];  // Max 32 warps per block (1024 threads / 32)
    
    size_t tid = threadIdx.x;
    size_t global_idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Each thread processes two elements for better efficiency
    float sum = 0.0f;
    if (global_idx < n) {
        sum = d_a[global_idx] * d_b[global_idx];
    }
    if (global_idx + blockDim.x < n) {
        sum += d_a[global_idx + blockDim.x] * d_b[global_idx + blockDim.x];
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
        sum = (tid < blockDim.x / 32) ? warp_sums[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    
    // Thread 0 atomically adds this block's result to the output
    if (tid == 0) {
        atomicAdd(d_output, sum);
    }
}

/*
 * Optimized Dot Product Kernel with vectorized loads (float4)
 * Uses 128-bit loads for better memory throughput
 * Requires n to be a multiple of 4 for best performance
 */
__global__ void dot_product_vectorized(float* d_output, float* d_a, float* d_b, size_t n) {
    __shared__ float warp_sums[32];
    
    size_t tid = threadIdx.x;
    size_t n4 = n / 4;  // Number of float4 elements
    
    // Cast to float4 pointers for vectorized access
    float4* a4 = reinterpret_cast<float4*>(d_a);
    float4* b4 = reinterpret_cast<float4*>(d_b);
    
    // Each block processes blockDim.x * 2 float4 elements
    size_t idx = blockIdx.x * blockDim.x * 2 + tid;
    
    float sum = 0.0f;
    
    // Load and compute products for first set of elements
    if (idx < n4) {
        float4 va = a4[idx];
        float4 vb = b4[idx];
        sum += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }
    
    // Load and compute products for second set of elements
    if (idx + blockDim.x < n4) {
        float4 va = a4[idx + blockDim.x];
        float4 vb = b4[idx + blockDim.x];
        sum += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }
    
    // Warp-level reduction
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warp_reduce_sum(sum);
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warp_sums[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    
    if (tid == 0) {
        atomicAdd(d_output, sum);
    }
}

/*
 * Main solve function
 * Computes dot product: result = sum(a[i] * b[i]) for i = 0 to n-1
 * 
 * Uses vectorized kernel for large arrays divisible by 4,
 * falls back to standard kernel for smaller or non-aligned arrays.
 */
void solve(float* d_a, float* d_b, float* d_output, size_t n) {
    if (n == 0) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));
        return;
    }
    
    // Initialize output to 0 (for atomic accumulation)
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));
    
    // Choose kernel based on array size and alignment
    if (n >= 1024 && n % 4 == 0) {
        // Use vectorized kernel for large, aligned arrays
        // Each thread processes 8 elements (2 * float4)
        size_t elements_per_block = BLOCK_SIZE * 8;
        int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        
        dot_product_vectorized<<<num_blocks, BLOCK_SIZE>>>(d_output, d_a, d_b, n);
    } else {
        // Standard kernel: each thread processes 2 elements
        size_t elements_per_block = BLOCK_SIZE * 2;
        int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        
        dot_product_kernel<<<num_blocks, BLOCK_SIZE>>>(d_output, d_a, d_b, n);
    }
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
