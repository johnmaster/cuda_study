#include "kernels.cuh"

void rand_init(half* h_data, size_t n) {
    std::mt19937 gen(200);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        // Generate random float and convert to half
        h_data[i] = __float2half(dist(gen));
    }
}

half* alloc_host(size_t n) {
    return new half[n];
}

half* alloc_device(size_t n) {
    half* p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(half)));
    return p;
}

void free_host(half* p) {
    delete[] p;
}

void free_device(half* p) {
    CHECK_CUDA(cudaFree(p));
}

/*
 * Warp-level reduction using shuffle instructions
 * Each thread in a warp exchanges values with other threads
 * and accumulates them to produce a partial sum
 * Uses FP32 for accumulation precision
 */
__device__ float warp_reduce_sum(float val) {
    // Use shuffle down to reduce within a warp (32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * FP16 Dot Product Kernel
 * 
 * Each thread computes the product of corresponding elements from vectors a and b,
 * then all products are reduced (summed) to produce the final dot product.
 * 
 * Key features:
 * - Input vectors use FP16 (half precision)
 * - Accumulation uses FP32 for better precision
 * - Final result is converted back to FP16
 * 
 * Strategy:
 * 1. Each thread loads FP16 elements, converts to FP32, and multiplies
 * 2. Use warp shuffle for fast intra-warp reduction (in FP32)
 * 3. Store warp results to shared memory
 * 4. Final reduction of warp results
 * 5. Atomic add to accumulate block results (FP32)
 */
__global__ void dot_product_fp16_kernel(float* d_partial_sum, 
                                         const half* __restrict__ d_a, 
                                         const half* __restrict__ d_b, 
                                         size_t n) {
    __shared__ float warp_sums[32];  // Max 32 warps per block (1024 threads / 32)
    
    size_t tid = threadIdx.x;
    size_t global_idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Each thread processes two elements for better efficiency
    // Accumulate in FP32 for precision
    float sum = 0.0f;
    if (global_idx < n) {
        // Load FP16 values, convert to FP32, multiply
        float a_val = __half2float(d_a[global_idx]);
        float b_val = __half2float(d_b[global_idx]);
        sum = a_val * b_val;
    }
    if (global_idx + blockDim.x < n) {
        float a_val = __half2float(d_a[global_idx + blockDim.x]);
        float b_val = __half2float(d_b[global_idx + blockDim.x]);
        sum += a_val * b_val;
    }
    
    // Warp-level reduction using shuffle (in FP32)
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
        atomicAdd(d_partial_sum, sum);
    }
}

/*
 * Optimized FP16 Dot Product Kernel with vectorized loads (half2)
 * Uses 32-bit loads (half2) for better memory throughput
 * Requires n to be a multiple of 2 for best performance
 * 
 * half2 allows loading two FP16 values at once and can use
 * native FP16 multiplication followed by FP32 accumulation
 */
__global__ void dot_product_fp16_vectorized(float* d_partial_sum, 
                                             const half* __restrict__ d_a, 
                                             const half* __restrict__ d_b, 
                                             size_t n) {
    __shared__ float warp_sums[32];
    
    size_t tid = threadIdx.x;
    size_t n2 = n / 2;  // Number of half2 elements
    
    // Cast to half2 pointers for vectorized access
    const half2* a2 = reinterpret_cast<const half2*>(d_a);
    const half2* b2 = reinterpret_cast<const half2*>(d_b);
    
    // Each block processes blockDim.x * 2 half2 elements
    size_t idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Accumulate in FP32 for precision
    float sum = 0.0f;
    
    // Load and compute products for first set of elements
    if (idx < n2) {
        half2 va = a2[idx];
        half2 vb = b2[idx];
        // Convert half2 to two floats, multiply and accumulate in FP32
        sum += __half2float(va.x) * __half2float(vb.x);
        sum += __half2float(va.y) * __half2float(vb.y);
    }
    
    // Load and compute products for second set of elements
    if (idx + blockDim.x < n2) {
        half2 va = a2[idx + blockDim.x];
        half2 vb = b2[idx + blockDim.x];
        sum += __half2float(va.x) * __half2float(vb.x);
        sum += __half2float(va.y) * __half2float(vb.y);
    }
    
    // Warp-level reduction (in FP32)
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
        atomicAdd(d_partial_sum, sum);
    }
}

/*
 * Kernel to convert the final FP32 result to FP16
 */
__global__ void convert_result_to_fp16(half* d_output, const float* d_input) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_output[0] = __float2half(d_input[0]);
    }
}

/*
 * Main solve function
 * Computes dot product: result = sum(a[i] * b[i]) for i = 0 to n-1
 * 
 * Input vectors are FP16 (half precision)
 * Accumulation is done in FP32 for numerical precision
 * Final result is stored as FP16 (half)
 * 
 * Uses vectorized kernel for large arrays divisible by 2,
 * falls back to standard kernel for smaller or non-aligned arrays.
 */
void solve(const half* d_a, const half* d_b, half* d_output, size_t n) {
    if (n == 0) {
        half zero = __float2half(0.0f);
        CHECK_CUDA(cudaMemcpy(d_output, &zero, sizeof(half), cudaMemcpyHostToDevice));
        return;
    }
    
    // Allocate temporary FP32 storage for accumulation
    float* d_fp32_result;
    CHECK_CUDA(cudaMalloc(&d_fp32_result, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_fp32_result, 0, sizeof(float)));
    
    // Choose kernel based on array size and alignment
    if (n >= 1024 && n % 2 == 0) {
        // Use vectorized kernel for large, aligned arrays
        // Each thread processes 4 half elements (2 * half2)
        size_t elements_per_block = BLOCK_SIZE * 4;
        int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        
        dot_product_fp16_vectorized<<<num_blocks, BLOCK_SIZE>>>(d_fp32_result, d_a, d_b, n);
    } else {
        // Standard kernel: each thread processes 2 elements
        size_t elements_per_block = BLOCK_SIZE * 2;
        int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        
        dot_product_fp16_kernel<<<num_blocks, BLOCK_SIZE>>>(d_fp32_result, d_a, d_b, n);
    }
    
    CHECK_CUDA(cudaGetLastError());
    
    // Convert FP32 result back to FP16
    convert_result_to_fp16<<<1, 1>>>(d_output, d_fp32_result);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Free temporary FP32 storage
    CHECK_CUDA(cudaFree(d_fp32_result));
}
