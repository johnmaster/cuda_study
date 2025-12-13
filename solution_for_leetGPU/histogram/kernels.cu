#include "kernels.cuh"

void rand_init(int32_t* h_data, size_t n, int32_t num_bins) {
    std::mt19937 gen(42);
    // Generate values mostly in valid range, with some out-of-range
    std::uniform_int_distribution<int32_t> dist(-10, num_bins + 10);
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

// Simple histogram kernel using global memory atomics
// Each thread processes multiple elements to improve efficiency
__global__ void histogramAtomicGlobal(const int32_t* __restrict__ input,
                                       int32_t* __restrict__ histogram,
                                       size_t N, int32_t num_bins) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = tid; i < N; i += stride) {
        int32_t val = input[i];
        // Only count values in valid range [0, num_bins)
        if (val >= 0 && val < num_bins) {
            atomicAdd(&histogram[val], 1);
        }
    }
}

// Optimized histogram kernel using shared memory
// First accumulates in per-block shared memory histogram, then merges to global
__global__ void histogramSharedMem(const int32_t* __restrict__ input,
                                    int32_t* __restrict__ histogram,
                                    size_t N, int32_t num_bins) {
    extern __shared__ int32_t s_hist[];
    
    // Initialize shared memory histogram to zero
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = tid; i < N; i += stride) {
        int32_t val = input[i];
        // Only count values in valid range [0, num_bins)
        if (val >= 0 && val < num_bins) {
            atomicAdd(&s_hist[val], 1);
        }
    }
    __syncthreads();
    
    // Merge shared memory histogram to global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (s_hist[i] > 0) {
            atomicAdd(&histogram[i], s_hist[i]);
        }
    }
}

// Kernel to zero out histogram array
__global__ void zeroHistogram(int32_t* histogram, int32_t num_bins) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < static_cast<size_t>(num_bins)) {
        histogram[tid] = 0;
    }
}

void solve(const int32_t* input, int32_t* histogram, size_t N, int32_t num_bins) {
    if (N == 0 || num_bins <= 0) {
        return;
    }
    
    // Zero out the histogram
    int zero_blocks = (num_bins + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zeroHistogram<<<zero_blocks, BLOCK_SIZE>>>(histogram, num_bins);
    
    // Calculate grid size - use enough blocks for good occupancy
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 256);  // Cap to avoid excessive blocks
    
    // Choose kernel based on num_bins size
    // Shared memory is efficient when histogram fits in shared memory
    // Maximum shared memory per block is typically 48KB = 12K int32_t values
    const int MAX_SHARED_BINS = 8192;  // Conservative limit
    
    if (num_bins <= MAX_SHARED_BINS) {
        // Use shared memory optimization
        size_t shared_mem_size = num_bins * sizeof(int32_t);
        histogramSharedMem<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
            input, histogram, N, num_bins);
    } else {
        // Fall back to global memory atomics for large histograms
        histogramAtomicGlobal<<<num_blocks, BLOCK_SIZE>>>(
            input, histogram, N, num_bins);
    }
}
