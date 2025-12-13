#include "kernels.cuh"

void rand_init(int32_t* h_data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-100, 100);
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

// Warp-level reduction to find maximum
__device__ __forceinline__ int32_t warpReduceMax(int32_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction to find maximum
__device__ __forceinline__ int32_t blockReduceMax(int32_t val) {
    __shared__ int32_t shared[32];
    
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    val = warpReduceMax(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : INT_MIN;
    if (warp_id == 0) {
        val = warpReduceMax(val);
    }
    
    return val;
}

// Standard sliding window kernel for small-medium window sizes
__global__ void maxWindowSumStandard(const int32_t* __restrict__ d_input, 
                                      int32_t* __restrict__ d_block_max,
                                      size_t N, size_t window_size,
                                      size_t windows_per_thread) {
    /*
    @num_windows: the number of windows in the input array
    @windows_per_thread: the number of windows each thread will process
    */
    const size_t num_windows = N - window_size + 1;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t start_window = tid * windows_per_thread;
    
    int32_t local_max = INT_MIN;
    
    if (start_window < num_windows) {
        int64_t window_sum = 0;
        size_t data_start = start_window;
        
        // Vectorized initial sum
        const int4* d_input4 = reinterpret_cast<const int4*>(d_input + data_start);
        size_t vec_count = window_size / 4;
        bool aligned = ((reinterpret_cast<uintptr_t>(d_input + data_start) & 15) == 0);
        
        size_t i = 0;
        if (aligned && vec_count > 0) {
            for (size_t v = 0; v < vec_count; v++) {
                int4 vals = d_input4[v];
                window_sum += vals.x + vals.y + vals.z + vals.w;
            }
            i = vec_count * 4;
        }
        for (; i < window_size; i++) {
            window_sum += d_input[data_start + i];
        }
        
        local_max = static_cast<int32_t>(window_sum);
        
        size_t end_window = min(start_window + windows_per_thread, num_windows);
        for (size_t w = start_window + 1; w < end_window; w++) {
            window_sum = window_sum - d_input[w - 1] + d_input[w + window_size - 1];
            local_max = max(local_max, static_cast<int32_t>(window_sum));
        }
    }
    
    int32_t block_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        d_block_max[blockIdx.x] = block_max;
    }
}

// Shared memory kernel for large window sizes
// Uses cooperative loading and shared memory prefix sum
// TILE_SIZE is the number of windows processed per block
#define TILE_SIZE 256
#define MAX_SHARED_ELEMENTS 4096  // Maximum elements in shared memory

__global__ void maxWindowSumSharedMem(const int32_t* __restrict__ d_input,
                                       int32_t* __restrict__ d_block_max,
                                       size_t N, size_t window_size) {
    extern __shared__ int32_t s_data[];  // Dynamic shared memory
    
    const size_t num_windows = N - window_size + 1;
    const size_t block_start = blockIdx.x * TILE_SIZE;  // First window this block handles
    
    if (block_start >= num_windows) {
        if (threadIdx.x == 0) {
            d_block_max[blockIdx.x] = INT_MIN;
        }
        return;
    }
    
    // Number of windows this block actually processes
    const size_t windows_this_block = min((size_t)TILE_SIZE, num_windows - block_start);
    // Number of elements we need: windows_this_block + window_size - 1
    const size_t elements_needed = windows_this_block + window_size - 1;
    const size_t data_start = block_start;
    
    // Cooperative loading: all threads load data into shared memory
    for (size_t i = threadIdx.x; i < elements_needed; i += blockDim.x) {
        if (data_start + i < N) {
            s_data[i] = d_input[data_start + i];
        }
    }
    __syncthreads();
    
    // Each thread computes one or more window sums
    int32_t local_max = INT_MIN;
    const size_t windows_per_thread = (windows_this_block + blockDim.x - 1) / blockDim.x;
    
    for (size_t w = 0; w < windows_per_thread; w++) {
        size_t local_window_idx = threadIdx.x + w * blockDim.x;
        if (local_window_idx < windows_this_block) {
            // Compute window sum from shared memory
            int64_t sum = 0;
            for (size_t i = 0; i < window_size; i++) {
                sum += s_data[local_window_idx + i];
            }
            local_max = max(local_max, static_cast<int32_t>(sum));
        }
    }
    
    int32_t block_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        d_block_max[blockIdx.x] = block_max;
    }
}

// Even more optimized: use sliding window within shared memory
__global__ void maxWindowSumSharedSliding(const int32_t* __restrict__ d_input,
                                           int32_t* __restrict__ d_block_max,
                                           size_t N, size_t window_size) {
    extern __shared__ int32_t s_data[];
    
    const size_t num_windows = N - window_size + 1;
    const size_t block_start = blockIdx.x * TILE_SIZE;
    
    if (block_start >= num_windows) {
        if (threadIdx.x == 0) {
            d_block_max[blockIdx.x] = INT_MIN;
        }
        return;
    }
    
    const size_t windows_this_block = min((size_t)TILE_SIZE, num_windows - block_start);
    const size_t elements_needed = windows_this_block + window_size - 1;
    const size_t data_start = block_start;
    
    // Cooperative loading into shared memory
    for (size_t i = threadIdx.x; i < elements_needed; i += blockDim.x) {
        if (data_start + i < N) {
            s_data[i] = d_input[data_start + i];
        }
    }
    __syncthreads();
    
    // Divide windows among threads, each thread uses sliding window
    const size_t total_threads = blockDim.x;
    const size_t windows_per_thread = (windows_this_block + total_threads - 1) / total_threads;
    const size_t my_start = threadIdx.x * windows_per_thread;
    const size_t my_end = min(my_start + windows_per_thread, windows_this_block);
    
    int32_t local_max = INT_MIN;
    
    if (my_start < windows_this_block) {
        // Compute first window sum
        int64_t window_sum = 0;
        for (size_t i = 0; i < window_size; i++) {
            window_sum += s_data[my_start + i];
        }
        local_max = static_cast<int32_t>(window_sum);
        
        // Slide through remaining windows
        for (size_t w = my_start + 1; w < my_end; w++) {
            window_sum = window_sum - s_data[w - 1] + s_data[w + window_size - 1];
            local_max = max(local_max, static_cast<int32_t>(window_sum));
        }
    }
    
    int32_t block_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        d_block_max[blockIdx.x] = block_max;
    }
}

// Final reduction kernel
__global__ void reduceBlockMax(const int32_t* __restrict__ d_block_max, 
                                int32_t* __restrict__ d_output, 
                                size_t num_blocks) {
    int32_t val = INT_MIN;
    
    for (size_t i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        val = max(val, d_block_max[i]);
    }
    
    val = blockReduceMax(val);
    
    if (threadIdx.x == 0) {
        *d_output = val;
    }
}

// Workspace
static int32_t* d_workspace = nullptr;
static size_t workspace_size = 0;

void solve(int32_t* d_input, int32_t* d_output, size_t N, size_t window_size) {
    if (window_size > N || window_size == 0) {
        int32_t min_val = INT_MIN;
        CHECK_CUDA(cudaMemcpy(d_output, &min_val, sizeof(int32_t), cudaMemcpyHostToDevice));
        return;
    }
    
    size_t num_windows = N - window_size + 1;
    
    // Use standard kernel with adaptive windows per thread
    // Key: windows_per_thread should be proportional to window_size to amortize initial sum cost
    size_t windows_per_thread;
    if (window_size < 64) {
        windows_per_thread = 32;
    } else if (window_size < 256) {
        windows_per_thread = 64;
    } else if (window_size < 1024) {
        windows_per_thread = 256;
    } else {
        windows_per_thread = 512;
    }
    
    // windows_per_thread is the number of windows each thread will process
    // num_tasks represents the number of threads
    size_t num_tasks = (num_windows + windows_per_thread - 1) / windows_per_thread;
    size_t numBlocks = (num_tasks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = max(numBlocks, (size_t)1);
    
    if (workspace_size < numBlocks) {
        if (d_workspace) cudaFree(d_workspace);
        workspace_size = max(numBlocks * 2, (size_t)256);
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size * sizeof(int32_t)));
    }
    
    maxWindowSumStandard<<<numBlocks, BLOCK_SIZE>>>(
        d_input, d_workspace, N, window_size, windows_per_thread);
    
    // Final reduction
    reduceBlockMax<<<1, BLOCK_SIZE>>>(d_workspace, d_output, numBlocks);
}

void cleanup_workspace() {
    if (d_workspace != nullptr) {
        cudaFree(d_workspace);
        d_workspace = nullptr;
        workspace_size = 0;
    }
}
