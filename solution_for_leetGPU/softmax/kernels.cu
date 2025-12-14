#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(42);
    // Generate values in a reasonable range to test numerical stability
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

// Block-level reduction for finding maximum
__device__ float blockReduceMax(float val) {
    __shared__ float shared[32];  // One value per warp
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // First reduce within warp
    val = warpReduceMax(val);
    
    // Write reduced value from each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Read from shared memory only if thread is in first warp
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -INFINITY;
    
    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceMax(val);
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

// Kernel to find the maximum value in the array
__global__ void findMaxKernel(const float* __restrict__ input, 
                               float* __restrict__ blockMax,
                               size_t N) {
    float maxVal = -INFINITY;
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Each thread finds max of its elements
    for (size_t i = tid; i < N; i += stride) {
        maxVal = fmaxf(maxVal, input[i]);
    }
    
    // Reduce within block
    maxVal = blockReduceMax(maxVal);
    
    // Write block result
    if (threadIdx.x == 0) {
        blockMax[blockIdx.x] = maxVal;
    }
}

// Kernel to reduce block maximums to final maximum
__global__ void reduceMaxKernel(float* __restrict__ blockMax, 
                                 int numBlocks) {
    float maxVal = -INFINITY;
    
    // Read all block maximums
    for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
        maxVal = fmaxf(maxVal, blockMax[i]);
    }
    
    // Reduce within block
    maxVal = blockReduceMax(maxVal);
    
    // Write final result to first element
    if (threadIdx.x == 0) {
        blockMax[0] = maxVal;
    }
}

// Kernel to compute exp(x - max) and find sum of exponentials
__global__ void expAndSumKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 float* __restrict__ blockSum,
                                 float maxVal,
                                 size_t N) {
    float sum = 0.0f;
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Each thread computes exp(x - max) for its elements and accumulates sum
    for (size_t i = tid; i < N; i += stride) {
        float expVal = expf(input[i] - maxVal);
        output[i] = expVal;
        sum += expVal;
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

// Kernel to normalize by dividing by sum
__global__ void normalizeKernel(float* __restrict__ output,
                                 float sum,
                                 size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float invSum = 1.0f / sum;
    
    for (size_t i = tid; i < N; i += stride) {
        output[i] *= invSum;
    }
}

// Single-block softmax kernel for small arrays (more efficient for small N)
__global__ void softmaxSingleBlockKernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          size_t N) {
    __shared__ float s_max;
    __shared__ float s_sum;
    
    float maxVal = -INFINITY;
    float sum = 0.0f;
    
    // Step 1: Find maximum (each thread handles multiple elements)
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        maxVal = fmaxf(maxVal, input[i]);
    }
    maxVal = blockReduceMax(maxVal);
    
    if (threadIdx.x == 0) {
        s_max = maxVal;
    }
    __syncthreads();
    maxVal = s_max;
    
    // Step 2: Compute exp(x - max) and sum
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float expVal = expf(input[i] - maxVal);
        output[i] = expVal;
        sum += expVal;
    }
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        s_sum = sum;
    }
    __syncthreads();
    sum = s_sum;
    
    // Step 3: Normalize
    float invSum = 1.0f / sum;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] *= invSum;
    }
}

void solve(const float* input, float* output, size_t N) {
    if (N == 0) {
        return;
    }
    
    // For small arrays, use single-block kernel
    if (N <= BLOCK_SIZE * 32) {
        softmaxSingleBlockKernel<<<1, BLOCK_SIZE>>>(input, output, N);
        return;
    }
    
    // For larger arrays, use multi-block approach
    int numBlocks = min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    
    // Allocate temporary storage for block results
    float* d_blockResults;
    CHECK_CUDA(cudaMalloc(&d_blockResults, numBlocks * sizeof(float)));
    
    // Step 1: Find maximum value
    findMaxKernel<<<numBlocks, BLOCK_SIZE>>>(input, d_blockResults, N);
    reduceMaxKernel<<<1, BLOCK_SIZE>>>(d_blockResults, numBlocks);
    
    // Get max value
    float maxVal;
    CHECK_CUDA(cudaMemcpy(&maxVal, d_blockResults, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Step 2: Compute exp(x - max) and sum
    expAndSumKernel<<<numBlocks, BLOCK_SIZE>>>(input, output, d_blockResults, maxVal, N);
    reduceSumKernel<<<1, BLOCK_SIZE>>>(d_blockResults, numBlocks);
    
    // Get sum value
    float sum;
    CHECK_CUDA(cudaMemcpy(&sum, d_blockResults, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Step 3: Normalize
    normalizeKernel<<<numBlocks, BLOCK_SIZE>>>(output, sum, N);
    
    // Free temporary storage
    CHECK_CUDA(cudaFree(d_blockResults));
}
