#include "kernels.cuh"

// ============================================
// Kernel 1: Naive Implementation
// Each thread processes one element, requires multiple passes
// ============================================
__global__ void layerNormNaiveKernel(const float* input, float* output,
                                      const float* gamma, const float* beta,
                                      int batch_size, int hidden_dim, float epsilon,
                                      float* mean_buffer, float* var_buffer) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (row >= batch_size || col >= hidden_dim) return;
    
    int idx = row * hidden_dim + col;
    
    // Read mean and variance (computed in separate kernels)
    float mean = mean_buffer[row];
    float var = var_buffer[row];
    
    // Normalize
    float normalized = (input[idx] - mean) / sqrtf(var + epsilon);
    
    // Apply affine transformation
    output[idx] = gamma[col] * normalized + beta[col];
}

__global__ void computeMeanKernel(const float* input, float* mean_buffer,
                                   int batch_size, int hidden_dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        sum += input[row * hidden_dim + i];
    }
    
    // Block reduction
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];    
            // Compute variance
            computeVarianceKernel<<<batch_size, 256>>>(input, mean_buffer, var_buffer, batch_size, hidden_dim);
            
        
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        mean_buffer[row] = shared_sum[0] / hidden_dim;
    }
}

__global__ void computeVarianceKernel(const float* input, const float* mean_buffer,
                                       float* var_buffer, int batch_size, int hidden_dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    float mean = mean_buffer[row];
    float sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float diff = input[row * hidden_dim + i] - mean;
        sum_sq += diff * diff;
    }
    
    // Block reduction
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        var_buffer[row] = shared_sum[0] / hidden_dim;
    }
}

void layerNormNaive(const float* input, float* output,
                    const float* gamma, const float* beta,
                    int batch_size, int hidden_dim, float epsilon) {
    float *mean_buffer, *var_buffer;
    CHECK_CUDA(cudaMalloc(&mean_buffer, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&var_buffer, batch_size * sizeof(float)));
    
    // Compute mean
    computeMeanKernel<<<batch_size, 256>>>(input, mean_buffer, batch_size, hidden_dim);
    
    // Compute variance
    computeVarianceKernel<<<batch_size, 256>>>(input, mean_buffer, var_buffer, batch_size, hidden_dim);
    
    // Apply normalization
    dim3 block(256);
    dim3 grid(batch_size, (hidden_dim + block.x - 1) / block.x);
    layerNormNaiveKernel<<<grid, block>>>(input, output, gamma, beta, batch_size, hidden_dim,
                                          epsilon, mean_buffer, var_buffer);
    
    CHECK_CUDA(cudaFree(mean_buffer));
    CHECK_CUDA(cudaFree(var_buffer));
}

// ============================================
// Kernel 2: Block-level Reduction
// Each block handles one row, compute mean and variance in one pass
// ============================================
__global__ void layerNormBlockReductionKernel(const float* input, float* output,
                                               const float* gamma, const float* beta,
                                               int batch_size, int hidden_dim, float epsilon) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    
    // Compute local sum and sum of squares
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        float val = input[row * hidden_dim + col];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    shared_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean = shared_sum[0] / hidden_dim;
    float variance = (shared_sum_sq[0] / hidden_dim) - (mean * mean);
    float inv_std = rsqrtf(variance + epsilon);
    
    // Apply normalization
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        int idx = row * hidden_dim + col;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = gamma[col] * normalized + beta[col];
    }
}

void layerNormBlockReduction(const float* input, float* output,
                              const float* gamma, const float* beta,
                              int batch_size, int hidden_dim, float epsilon) {
    int threads = 256;
    layerNormBlockReductionKernel<<<batch_size, threads>>>(input, output, gamma, beta,
                                                            batch_size, hidden_dim, epsilon);
}

// ============================================
// Kernel 3: Warp-level Reduction using Shuffle
// ============================================
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layerNormWarpReductionKernel(const float* input, float* output,
                                              const float* gamma, const float* beta,
                                              int batch_size, int hidden_dim, float epsilon) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int WARP_SIZE = 32;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    __shared__ float warp_sum[8];  // Max 8 warps per block
    __shared__ float warp_sum_sq[8];
    
    // Compute local sum and sum of squares
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        float val = input[row * hidden_dim + col];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp-level reduction
    local_sum = warpReduceSum(local_sum);
    local_sum_sq = warpReduceSum(local_sum_sq);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        local_sum = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        local_sum_sq = (lane < num_warps) ? warp_sum_sq[lane] : 0.0f;
        
        local_sum = warpReduceSum(local_sum);
        local_sum_sq = warpReduceSum(local_sum_sq);
        
        if (lane == 0) {
            warp_sum[0] = local_sum;
            warp_sum_sq[0] = local_sum_sq;
        }
    }
    __syncthreads();
    
    // Compute mean and variance
    float mean = warp_sum[0] / hidden_dim;
    float variance = (warp_sum_sq[0] / hidden_dim) - (mean * mean);
    float inv_std = rsqrtf(variance + epsilon);
    
    // Apply normalization
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        int idx = row * hidden_dim + col;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = gamma[col] * normalized + beta[col];
    }
}

void layerNormWarpReduction(const float* input, float* output,
                            const float* gamma, const float* beta,
                            int batch_size, int hidden_dim, float epsilon) {
    int threads = 256;
    layerNormWarpReductionKernel<<<batch_size, threads>>>(input, output, gamma, beta,
                                                           batch_size, hidden_dim, epsilon);
}

// ============================================
// Kernel 4: Vectorized Load/Store
// Use float4 for memory coalescing
// ============================================
__global__ void layerNormVectorizedKernel(const float* input, float* output,
                                           const float* gamma, const float* beta,
                                           int batch_size, int hidden_dim, float epsilon) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    
    const int VEC_SIZE = 4;
    const float4* input4 = reinterpret_cast<const float4*>(input + row * hidden_dim);
    float4* output4 = reinterpret_cast<float4*>(output + row * hidden_dim);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    
    int hidden_dim4 = hidden_dim / VEC_SIZE;
    
    // Compute local sum and sum of squares using vectorized loads
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_dim4; i += blockDim.x) {
        float4 val = input4[i];
        local_sum += val.x + val.y + val.z + val.w;
        local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Handle remaining elements
    for (int i = hidden_dim4 * VEC_SIZE + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = input[row * hidden_dim + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    shared_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean = shared_sum[0] / hidden_dim;
    float variance = (shared_sum_sq[0] / hidden_dim) - (mean * mean);
    float inv_std = rsqrtf(variance + epsilon);
    
    // Apply normalization using vectorized loads/stores
    for (int i = threadIdx.x; i < hidden_dim4; i += blockDim.x) {
        float4 val = input4[i];
        float4 g = gamma4[i];
        float4 b = beta4[i];
        
        float4 result;
        result.x = g.x * (val.x - mean) * inv_std + b.x;
        result.y = g.y * (val.y - mean) * inv_std + b.y;
        result.z = g.z * (val.z - mean) * inv_std + b.z;
        result.w = g.w * (val.w - mean) * inv_std + b.w;
        
        output4[i] = result;
    }
    
    // Handle remaining elements
    for (int i = hidden_dim4 * VEC_SIZE + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        int idx = row * hidden_dim + i;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

void layerNormVectorized(const float* input, float* output,
                         const float* gamma, const float* beta,
                         int batch_size, int hidden_dim, float epsilon) {
    int threads = 256;
    layerNormVectorizedKernel<<<batch_size, threads>>>(input, output, gamma, beta,
                                                        batch_size, hidden_dim, epsilon);
}

// ============================================
// Kernel 5: Welford's Online Algorithm
// Better numerical stability for variance computation
// ============================================
__global__ void layerNormWelfordKernel(const float* input, float* output,
                                        const float* gamma, const float* beta,
                                        int batch_size, int hidden_dim, float epsilon) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    __shared__ float shared_mean[256];
    __shared__ float shared_m2[256];
    __shared__ int shared_count[256];
    
    // Welford's online algorithm for each thread
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        float val = input[row * hidden_dim + col];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    shared_mean[threadIdx.x] = mean;
    shared_m2[threadIdx.x] = m2;
    shared_count[threadIdx.x] = count;
    __syncthreads();
    
    // Merge Welford states across threads
    if (threadIdx.x == 0) {
        float global_mean = 0.0f;
        float global_m2 = 0.0f;
        int global_count = 0;
        
        for (int i = 0; i < blockDim.x; i++) {
            if (shared_count[i] == 0) continue;
            
            int count_a = global_count;
            int count_b = shared_count[i];
            int new_count = count_a + count_b;
            
            if (new_count == 0) continue;
            
            float mean_a = global_mean;
            float mean_b = shared_mean[i];
            float m2_a = global_m2;
            float m2_b = shared_m2[i];
            
            float delta = mean_b - mean_a;
            global_mean = (count_a * mean_a + count_b * mean_b) / new_count;
            global_m2 = m2_a + m2_b + delta * delta * count_a * count_b / new_count;
            global_count = new_count;
        }
        
        shared_mean[0] = global_mean;
        shared_m2[0] = global_m2 / global_count;  // variance
    }
    __syncthreads();
    
    float mean_val = shared_mean[0];
    float variance = shared_m2[0];
    float inv_std = rsqrtf(variance + epsilon);
    
    // Apply normalization
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        int idx = row * hidden_dim + col;
        float normalized = (input[idx] - mean_val) * inv_std;
        output[idx] = gamma[col] * normalized + beta[col];
    }
}

void layerNormWelford(const float* input, float* output,
                      const float* gamma, const float* beta,
                      int batch_size, int hidden_dim, float epsilon) {
    int threads = 256;
    layerNormWelfordKernel<<<batch_size, threads>>>(input, output, gamma, beta,
                                                     batch_size, hidden_dim, epsilon);
}

// ============================================
// CPU Reference Implementation
// ============================================
void layerNormCPU(const float* input, float* output,
                  const float* gamma, const float* beta,
                  int batch_size, int hidden_dim, float epsilon) {
    for (int b = 0; b < batch_size; b++) {
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            mean += input[b * hidden_dim + i];
        }
        mean /= hidden_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float diff = input[b * hidden_dim + i] - mean;
            variance += diff * diff;
        }
        variance /= hidden_dim;
        
        // Normalize and apply affine transformation
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int i = 0; i < hidden_dim; i++) {
            int idx = b * hidden_dim + i;
            float normalized = (input[idx] - mean) * inv_std;
            output[idx] = gamma[i] * normalized + beta[i];
        }
    }
}

// ============================================
// Benchmark Function
// ============================================
void benchMark(int batch_size, int hidden_dim) {
    std::cout << "\n========================================\n";
    std::cout << "LayerNorm: [" << batch_size << " x " << hidden_dim << "]\n";
    std::cout << "========================================\n";
    
    float epsilon = 1e-5f;
    size_t data_size = batch_size * hidden_dim;
    size_t bytes = data_size * sizeof(float);
    size_t param_bytes = hidden_dim * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_input(data_size);
    std::vector<float> h_output(data_size);
    std::vector<float> h_output_cpu(data_size);
    std::vector<float> h_gamma(hidden_dim);
    std::vector<float> h_beta(hidden_dim);
    
    // Initialize input data
    for (size_t i = 0; i < data_size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    
    // Initialize gamma and beta
    for (int i = 0; i < hidden_dim; i++) {
        h_gamma[i] = 1.0f;  // Scale
        h_beta[i] = 0.0f;   // Shift
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma, param_bytes));
    CHECK_CUDA(cudaMalloc(&d_beta, param_bytes));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), param_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), param_bytes, cudaMemcpyHostToDevice));
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    layerNormCPU(h_input.data(), h_output_cpu.data(), h_gamma.data(), h_beta.data(),
                 batch_size, hidden_dim, epsilon);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // Benchmark configurations
    struct KernelConfig {
        std::string name;
        void (*func)(const float*, float*, const float*, const float*, int, int, float);
    };
    
    std::vector<KernelConfig> kernels = {
        {"Naive (3-pass)", layerNormNaive},
        {"Block Reduction", layerNormBlockReduction},
        {"Warp Reduction", layerNormWarpReduction},
        {"Vectorized", layerNormVectorized},
        {"Welford's Algorithm", layerNormWelford}
    };
    
    printf("\n%-25s %10s %12s %10s\n", "Kernel", "Time(ms)", "Bandwidth", "Error");
    printf("%-25s %10s %12s %10s\n", "------", "--------", "---------", "-----");
    printf("%-25s %10.3f %12s %10s\n", "CPU", cpu_time, "-", "-");
    
    for (auto& kernel : kernels) {
        // Warmup
        kernel.func(d_input, d_output, d_gamma, d_beta, batch_size, hidden_dim, epsilon);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        const int num_runs = 100;
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < num_runs; i++) {
            kernel.func(d_input, d_output, d_gamma, d_beta, batch_size, hidden_dim, epsilon);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        float avg_time = milliseconds / num_runs;
        
        // Copy result back
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Compute error
        float max_error = 0.0f;
        for (size_t i = 0; i < data_size; i++) {
            float error = std::abs(h_output[i] - h_output_cpu[i]);
            max_error = std::max(max_error, error);
        }
        
        // Compute bandwidth (read input + write output + read gamma + read beta)
        double bandwidth = (bytes * 2 + param_bytes * 2) / (avg_time / 1000.0) / 1e9;
        
        printf("%-25s %10.3f %10.2f GB/s %10.2e\n",
               kernel.name.c_str(), avg_time, bandwidth, max_error);
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
}

