#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
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

// ============================================================================
// GELU Activation Function Implementations
// ============================================================================
// GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
// 
// Two common implementations:
// 1. Exact: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// 2. Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// ============================================================================

// Device function for exact GELU using erf
__device__ __forceinline__ float gelu_exact(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    constexpr float SQRT_2_INV = 0.7071067811865475f;  // 1/sqrt(2)
    return x * 0.5f * (1.0f + erff(x * SQRT_2_INV));
}

// Device function for GELU using tanh approximation (faster)
__device__ __forceinline__ float gelu_tanh(float x) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ============================================================================
// Kernel 1: Naive GELU - one element per thread
// ============================================================================
__global__ void geluNaiveKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = gelu_exact(input[idx]);
    }
}

// ============================================================================
// Kernel 2: GELU with grid stride loop
// ============================================================================
__global__ void geluStrideKernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < N; i += stride) {
        output[i] = gelu_exact(input[i]);
    }
}

// ============================================================================
// Kernel 3: Vectorized GELU using float4
// ============================================================================
__global__ void geluVectorizedKernel(const float4* __restrict__ input,
                                      float4* __restrict__ output,
                                      size_t N4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < N4; i += stride) {
        float4 in = input[i];
        float4 out;
        out.x = gelu_exact(in.x);
        out.y = gelu_exact(in.y);
        out.z = gelu_exact(in.z);
        out.w = gelu_exact(in.w);
        output[i] = out;
    }
}

// Handle remaining elements
__global__ void geluRemainderKernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     size_t start,
                                     size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < N) {
        output[idx] = gelu_exact(input[idx]);
    }
}

// ============================================================================
// Kernel 4: GELU with tanh approximation (faster, slightly less accurate)
// ============================================================================
__global__ void geluTanhKernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < N; i += stride) {
        output[i] = gelu_tanh(input[i]);
    }
}

// ============================================================================
// Kernel 5: Vectorized GELU with tanh approximation
// ============================================================================
__global__ void geluTanhVectorizedKernel(const float4* __restrict__ input,
                                          float4* __restrict__ output,
                                          size_t N4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < N4; i += stride) {
        float4 in = input[i];
        float4 out;
        out.x = gelu_tanh(in.x);
        out.y = gelu_tanh(in.y);
        out.z = gelu_tanh(in.z);
        out.w = gelu_tanh(in.w);
        output[i] = out;
    }
}

__global__ void geluTanhRemainderKernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         size_t start,
                                         size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < N) {
        output[idx] = gelu_tanh(input[idx]);
    }
}

// ============================================================================
// Kernel 6: Fused vectorized load + unrolled processing
// ============================================================================
__global__ void geluUnrolledKernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t stride = blockDim.x * gridDim.x * 4;
    
    for (size_t i = idx; i + 3 < N; i += stride) {
        float v0 = input[i];
        float v1 = input[i + 1];
        float v2 = input[i + 2];
        float v3 = input[i + 3];
        
        output[i] = gelu_exact(v0);
        output[i + 1] = gelu_exact(v1);
        output[i + 2] = gelu_exact(v2);
        output[i + 3] = gelu_exact(v3);
    }
    
    // Handle remainder
    size_t remaining_start = (N / 4) * 4;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N - remaining_start) {
        size_t rem_idx = remaining_start + tid;
        output[rem_idx] = gelu_exact(input[rem_idx]);
    }
}

// ============================================================================
// Main solve function using vectorized kernel (best performance)
// ============================================================================
void solve(const float* input, float* output, size_t N) {
    if (N == 0) {
        return;
    }
    
    // Use vectorized kernel for best performance
    size_t N4 = N / 4;
    size_t remainder_start = N4 * 4;
    size_t remainder = N - remainder_start;
    
    int numBlocks = std::min((int)((N4 + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    
    if (N4 > 0) {
        geluVectorizedKernel<<<numBlocks, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output),
            N4
        );
    }
    
    // Handle remaining elements
    if (remainder > 0) {
        int remBlocks = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
        geluRemainderKernel<<<remBlocks, BLOCK_SIZE>>>(input, output, remainder_start, N);
    }
}

// ============================================================================
// Alternative solve functions for benchmarking different kernels
// ============================================================================

void solve_naive(const float* input, float* output, size_t N) {
    if (N == 0) return;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geluNaiveKernel<<<numBlocks, BLOCK_SIZE>>>(input, output, N);
}

void solve_stride(const float* input, float* output, size_t N) {
    if (N == 0) return;
    int numBlocks = std::min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    geluStrideKernel<<<numBlocks, BLOCK_SIZE>>>(input, output, N);
}

void solve_tanh(const float* input, float* output, size_t N) {
    if (N == 0) return;
    
    size_t N4 = N / 4;
    size_t remainder_start = N4 * 4;
    size_t remainder = N - remainder_start;
    
    int numBlocks = std::min((int)((N4 + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    
    if (N4 > 0) {
        geluTanhVectorizedKernel<<<numBlocks, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output),
            N4
        );
    }
    
    if (remainder > 0) {
        int remBlocks = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
        geluTanhRemainderKernel<<<remBlocks, BLOCK_SIZE>>>(input, output, remainder_start, N);
    }
}

void solve_unrolled(const float* input, float* output, size_t N) {
    if (N == 0) return;
    int numBlocks = std::min((int)((N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    geluUnrolledKernel<<<numBlocks, BLOCK_SIZE>>>(input, output, N);
}

