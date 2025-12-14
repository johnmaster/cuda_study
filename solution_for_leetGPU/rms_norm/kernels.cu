#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
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

// Warp-level reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // One value per warp
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // First reduce within warp
    val = warpReduceSum(val);
    
    // Write reduced value from each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Read from shared memory only if thread is in first warp
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    
    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Kernel to compute sum of squares
__global__ void sumSquaresKernel(const float* __restrict__ input,
                                  float* __restrict__ blockSum,
                                  size_t N) {
    float sum = 0.0f;
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Each thread computes sum of squares of its elements
    for (size_t i = tid; i < N; i += stride) {
        float val = input[i];
        sum += val * val;
    }
    
    // Reduce within block
    sum = blockReduceSum(sum);
    
    // Write block result
    if (threadIdx.x == 0) {
        blockSum[blockIdx.x] = sum;
    }
}

// Kernel to reduce block sums to final sum
__global__ void reduceSumKernel(float* __restrict__ blockSum, 
                                 int numBlocks) {
    float sum = 0.0f;
    
    // Read all block sums
    for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
        sum += blockSum[i];
    }
    
    // Reduce within block
    sum = blockReduceSum(sum);
    
    // Write final result to first element
    if (threadIdx.x == 0) {
        blockSum[0] = sum;
    }
}

// Kernel to apply RMS normalization
__global__ void rmsNormKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               float rms_inv,
                               float gamma,
                               float beta,
                               size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = tid; i < N; i += stride) {
        output[i] = gamma * (input[i] * rms_inv) + beta;
    }
}

// Single-block RMS norm kernel for small arrays (more efficient for small N)
__global__ void rmsNormSingleBlockKernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          float gamma,
                                          float beta,
                                          size_t N,
                                          float eps) {
    __shared__ float s_sumSq;
    
    float sumSq = 0.0f;
    
    // Step 1: Compute sum of squares (each thread handles multiple elements)
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float val = input[i];
        sumSq += val * val;
    }
    sumSq = blockReduceSum(sumSq);
    
    if (threadIdx.x == 0) {
        // Compute RMS = sqrt(mean(x^2) + eps)
        float meanSq = sumSq / static_cast<float>(N);
        s_sumSq = rsqrtf(meanSq + eps);  // Store 1/RMS for efficiency
    }
    __syncthreads();
    
    float rms_inv = s_sumSq;
    
    // Step 2: Apply normalization
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = gamma * (input[i] * rms_inv) + beta;
    }
}

void solve(const float* input, float* output, float gamma, float beta, size_t N, float eps) {
    if (N == 0) {
        return;
    }
    
    // For small arrays, use single-block kernel
    if (N <= BLOCK_SIZE * 32) {
        rmsNormSingleBlockKernel<<<1, BLOCK_SIZE>>>(input, output, gamma, beta, N, eps);
        return;
    }
    
    // For larger arrays, use multi-block approach
    int numBlocks = min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    
    // Allocate temporary storage for block results
    float* d_blockResults;
    CHECK_CUDA(cudaMalloc(&d_blockResults, numBlocks * sizeof(float)));
    
    // Step 1: Compute sum of squares
    sumSquaresKernel<<<numBlocks, BLOCK_SIZE>>>(input, d_blockResults, N);
    reduceSumKernel<<<1, BLOCK_SIZE>>>(d_blockResults, numBlocks);
    
    // Get sum of squares
    float sumSq;
    CHECK_CUDA(cudaMemcpy(&sumSq, d_blockResults, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute RMS and its inverse
    float meanSq = sumSq / static_cast<float>(N);
    float rms_inv = 1.0f / sqrtf(meanSq + eps);
    
    // Step 2: Apply normalization
    rmsNormKernel<<<numBlocks, BLOCK_SIZE>>>(input, output, rms_inv, gamma, beta, N);
    
    // Free temporary storage
    CHECK_CUDA(cudaFree(d_blockResults));
}
