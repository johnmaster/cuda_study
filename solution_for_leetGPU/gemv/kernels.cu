#include "kernels.cuh"

// ============================================================================
// Helper Functions
// ============================================================================

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

void rand_init(float* h_data, size_t n, unsigned seed = 0x2025) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i)
        h_data[i] = dist(rng);
}

float max_error(const float* ref, const float* test, size_t n) {
    float err = 0.0f;
    for (size_t i = 0; i < n; i++)
        err = std::max(err, std::abs(test[i] - ref[i]));
    return err;
}

// ============================================================================
// cuBLAS Reference
// ============================================================================

double run_cublas(float* dA, float* dx, float* dy, int M, int K, int repeats = 50) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    // warmup
    for (int i = 0; i < 10; ++i)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, K, M, &alpha, dA, K, dx, 1, &beta, dy, 1));

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeats; ++i)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, K, M, &alpha, dA, K, dx, 1, &beta, dy, 1));

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CHECK_CUBLAS(cublasDestroy(handle));

    return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

void printMatrixInformation(int M, int K) {
    std::cout << "\n";
    std::cout << "┌" << std::string(70, '-') << "┐\n";
    std::cout << "│" 
              << std::setw(70) << std::left 
              << (" GEMV: M = " + std::to_string(M) + ", K = " + std::to_string(K) + 
                  " | A[" + std::to_string(M) + "x" + std::to_string(K) + 
                  "] * x[" + std::to_string(K) + "] = y[" + std::to_string(M) + "] ").c_str()
              << "│\n";
    std::cout << "└" << std::string(70, '-') << "┘\n\n";

    std::cout << std::left
              << std::setw(45) << "Kernel"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GB/s"
              << std::setw(14) << "Max Error"
              << "\n";
    std::cout << std::string(80, '-') << "\n";
}

// ============================================================================
// Kernel 1: Naive GEMV (one thread per row)
// Each thread computes one element of the output vector
// ============================================================================

__global__ void sgemv_naive_kernel(float* A, float* x, float* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * x[k];
        }
        y[row] = sum;
    }
}

void sgemv_naive(float* A, float* x, float* y, int M, int K) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemv_naive_kernel<<<grid, block>>>(A, x, y, M, K);
}

// ============================================================================
// Kernel 2: Vectorized loads (float4)
// Uses 128-bit loads for better memory throughput
// ============================================================================

__global__ void sgemv_float4_kernel(float* A, float* x, float* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        int k = 0;
        
        // Vectorized loads (process 4 elements at a time)
        float4* A_row_f4 = reinterpret_cast<float4*>(A + row * K);
        float4* x_f4 = reinterpret_cast<float4*>(x);
        
        for (; k + 3 < K; k += 4) {
            float4 a4 = A_row_f4[k / 4];
            float4 x4 = x_f4[k / 4];
            sum += a4.x * x4.x + a4.y * x4.y + a4.z * x4.z + a4.w * x4.w;
        }
        
        // Handle remaining elements
        for (; k < K; ++k) {
            sum += A[row * K + k] * x[k];
        }
        
        y[row] = sum;
    }
}

void sgemv_float4(float* A, float* x, float* y, int M, int K) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemv_float4_kernel<<<grid, block>>>(A, x, y, M, K);
}

// ============================================================================
// Kernel 3: Warp-level reduction (one warp per row)
// Each warp cooperatively computes one output element
// Better for large K values
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sgemv_warp_kernel(float* A, float* x, float* y, int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id < M) {
        float sum = 0.0f;
        
        // Each thread in warp processes multiple elements
        for (int k = lane_id; k < K; k += WARP_SIZE) {
            sum += A[warp_id * K + k] * x[k];
        }
        
        // Warp-level reduction
        sum = warp_reduce_sum(sum);
        
        // First thread in warp writes result
        if (lane_id == 0) {
            y[warp_id] = sum;
        }
    }
}

void sgemv_warp(float* A, float* x, float* y, int M, int K) {
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int num_blocks = (M + warps_per_block - 1) / warps_per_block;
    sgemv_warp_kernel<<<num_blocks, BLOCK_SIZE>>>(A, x, y, M, K);
}

// ============================================================================
// Kernel 4: Warp-level with vectorized loads
// Combines warp cooperation with float4 vectorized memory access
// ============================================================================

__global__ void sgemv_warp_float4_kernel(float* A, float* x, float* y, int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id < M) {
        float sum = 0.0f;
        float4* A_row_f4 = reinterpret_cast<float4*>(A + warp_id * K);
        float4* x_f4 = reinterpret_cast<float4*>(x);
        int K4 = K / 4;
        
        // Each thread processes multiple float4 elements
        for (int k4 = lane_id; k4 < K4; k4 += WARP_SIZE) {
            float4 a4 = A_row_f4[k4];
            float4 x4 = x_f4[k4];
            sum += a4.x * x4.x + a4.y * x4.y + a4.z * x4.z + a4.w * x4.w;
        }
        
        // Handle remaining elements (if K is not divisible by 4)
        if (lane_id == 0) {
            for (int k = K4 * 4; k < K; ++k) {
                sum += A[warp_id * K + k] * x[k];
            }
        }
        
        // Warp-level reduction
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            y[warp_id] = sum;
        }
    }
}

void sgemv_warp_float4(float* A, float* x, float* y, int M, int K) {
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int num_blocks = (M + warps_per_block - 1) / warps_per_block;
    sgemv_warp_float4_kernel<<<num_blocks, BLOCK_SIZE>>>(A, x, y, M, K);
}

// ============================================================================
// Kernel 5: Block-level reduction with shared memory
// One block per row, threads cooperate via shared memory
// Good for very large K values
// ============================================================================

template <int BLOCK = 256>
__global__ void sgemv_block_kernel(float* A, float* x, float* y, int M, int K) {
    __shared__ float smem[BLOCK];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        
        // Each thread sums its portion
        for (int k = tid; k < K; k += BLOCK) {
            sum += A[row * K + k] * x[k];
        }
        
        smem[tid] = sum;
        __syncthreads();
        
        // Block-level reduction in shared memory
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            y[row] = smem[0];
        }
    }
}

void sgemv_block(float* A, float* x, float* y, int M, int K) {
    sgemv_block_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(A, x, y, M, K);
}

// ============================================================================
// Kernel 6: Block-level with float4 and warp shuffle
// Combines all optimizations for best performance
// ============================================================================

template <int BLOCK = 256>
__global__ void sgemv_block_float4_warp_kernel(float* A, float* x, float* y, int M, int K) {
    __shared__ float warp_sums[BLOCK / WARP_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    if (row < M) {
        float sum = 0.0f;
        
        float4* A_row_f4 = reinterpret_cast<float4*>(A + row * K);
        float4* x_f4 = reinterpret_cast<float4*>(x);
        int K4 = K / 4;
        
        // Vectorized accumulation
        for (int k4 = tid; k4 < K4; k4 += BLOCK) {
            float4 a4 = A_row_f4[k4];
            float4 x4 = x_f4[k4];
            sum += a4.x * x4.x + a4.y * x4.y + a4.z * x4.z + a4.w * x4.w;
        }
        
        // Handle remaining elements
        if (tid == 0) {
            for (int k = K4 * 4; k < K; ++k) {
                sum += A[row * K + k] * x[k];
            }
        }
        
        // Warp-level reduction
        sum = warp_reduce_sum(sum);
        
        // Store warp results to shared memory
        if (lane_id == 0) {
            warp_sums[warp_id] = sum;
        }
        __syncthreads();
        
        // Final reduction by first warp
        if (tid < BLOCK / WARP_SIZE) {
            sum = warp_sums[tid];
            sum = warp_reduce_sum(sum);
            
            if (tid == 0) {
                y[row] = sum;
            }
        }
    }
}

void sgemv_block_float4_warp(float* A, float* x, float* y, int M, int K) {
    sgemv_block_float4_warp_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(A, x, y, M, K);
}

// ============================================================================
// Kernel 7: Multi-row per block with vectorized loads
// Multiple rows processed per block for better occupancy on small M
// ============================================================================

template <int ROWS_PER_BLOCK = 4, int BLOCK = 256>
__global__ void sgemv_multi_row_kernel(float* A, float* x, float* y, int M, int K) {
    // ROWS_PER_BLOCK rows per block, each row processed by BLOCK/ROWS_PER_BLOCK threads
    constexpr int THREADS_PER_ROW = BLOCK / ROWS_PER_BLOCK;  // 64 threads per row
    
    __shared__ float smem[ROWS_PER_BLOCK][THREADS_PER_ROW];
    
    int base_row = blockIdx.x * ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int row_in_block = tid / THREADS_PER_ROW;
    int tid_in_row = tid % THREADS_PER_ROW;
    int row = base_row + row_in_block;
    
    float sum = 0.0f;
    
    if (row < M) {
        // Each thread accumulates its portion
        for (int k = tid_in_row; k < K; k += THREADS_PER_ROW) {
            sum += A[row * K + k] * x[k];
        }
    }
    
    // Store to shared memory for reduction
    smem[row_in_block][tid_in_row] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = THREADS_PER_ROW / 2; s > 0; s >>= 1) {
        if (tid_in_row < s) {
            smem[row_in_block][tid_in_row] += smem[row_in_block][tid_in_row + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid_in_row == 0 && row < M) {
        y[row] = smem[row_in_block][0];
    }
}

void sgemv_multi_row(float* A, float* x, float* y, int M, int K) {
    constexpr int ROWS_PER_BLOCK = 4;
    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    sgemv_multi_row_kernel<ROWS_PER_BLOCK, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(A, x, y, M, K);
}

// ============================================================================
// Benchmark Function
// ============================================================================

void benchMark(int M, int K) {
    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeX = K;
    const size_t sizeY = M;
    
    float* host_A = alloc_host(sizeA);
    float* host_x = alloc_host(sizeX);
    float* host_y_ref = alloc_host(sizeY);
    float* host_y_test = alloc_host(sizeY);

    rand_init(host_A, sizeA, 0x2025);
    rand_init(host_x, sizeX, 0x1234);

    float* device_A = alloc_device(sizeA);
    float* device_x = alloc_device(sizeX);
    float* device_y = alloc_device(sizeY);

    CHECK_CUDA(cudaMemcpy(device_A, host_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_x, host_x, sizeX * sizeof(float), cudaMemcpyHostToDevice));

    // Run cuBLAS reference
    CHECK_CUDA(cudaMemset(device_y, 0, sizeY * sizeof(float)));
    double cublas_ms = run_cublas(device_A, device_x, device_y, M, K, 50);
    CHECK_CUDA(cudaMemcpy(host_y_ref, device_y, sizeY * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate bandwidth (read A, read x, write y)
    double bytes = static_cast<double>(sizeA + sizeX + sizeY) * sizeof(float);
    double cublas_gbps = bytes / (cublas_ms * 1e-3) / 1e9;

    printMatrixInformation(M, K);

    std::cout << std::left
              << std::setw(45) << "cuBLAS (reference)"
              << std::setw(12) << std::fixed << std::setprecision(4) << cublas_ms
              << std::setw(12) << std::setprecision(2) << cublas_gbps
              << std::setw(14) << "0.0"
              << "\n";

    std::vector<Kernel> kernels = {
        {"sgemv_naive",                 sgemv_naive},
        {"sgemv_float4",                sgemv_float4},
        {"sgemv_warp",                  sgemv_warp},
        {"sgemv_warp_float4",           sgemv_warp_float4},
        {"sgemv_block",                 sgemv_block},
        {"sgemv_block_float4_warp",     sgemv_block_float4_warp},
        {"sgemv_multi_row",             sgemv_multi_row},
    };

    for (const auto& k : kernels) {
        CHECK_CUDA(cudaMemset(device_y, 0, sizeY * sizeof(float)));
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            k.func(device_A, device_x, device_y, M, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; ++i) {
            k.func(device_A, device_x, device_y, M, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count() / 50;
        double gbps = bytes / (ms * 1e-3) / 1e9;

        // Check error
        CHECK_CUDA(cudaMemcpy(host_y_test, device_y, sizeY * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = max_error(host_y_ref, host_y_test, sizeY);

        std::cout << std::left << std::setw(45) << k.name
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << ms
                  << std::setw(12) << std::setprecision(2) << gbps
                  << std::setw(14) << std::scientific << std::setprecision(2) << max_err 
                  << std::fixed << "\n";
    }

    free_host(host_A);
    free_host(host_x);
    free_host(host_y_ref);
    free_host(host_y_test);
    free_device(device_A);
    free_device(device_x);
    free_device(device_y);
}

