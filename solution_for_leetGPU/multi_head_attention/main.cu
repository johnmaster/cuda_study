#include "kernels.cuh"

// CPU reference implementation of multi-head self-attention
void cpu_multi_head_attention(
    const float* Q, const float* K, const float* V, float* output,
    int B, int N, int d, int num_heads
) {
    int d_h = d / num_heads;
    float scale = 1.0f / sqrtf((float)d_h);
    
    // Temporary storage for attention scores and weights
    std::vector<float> S(N * N);
    std::vector<float> A(N * N);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Compute S = Q_h @ K_h^T / sqrt(d_h)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < d_h; k++) {
                        float q_val = Q[b * N * d + i * d + h * d_h + k];
                        float k_val = K[b * N * d + j * d + h * d_h + k];
                        sum += q_val * k_val;
                    }
                    S[i * N + j] = sum * scale;
                }
            }
            
            // Apply row-wise softmax to get A
            for (int i = 0; i < N; i++) {
                // Find row max
                float max_val = S[i * N];
                for (int j = 1; j < N; j++) {
                    max_val = std::max(max_val, S[i * N + j]);
                }
                
                // Compute exp and sum
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    A[i * N + j] = std::exp(S[i * N + j] - max_val);
                    sum += A[i * N + j];
                }
                
                // Normalize
                for (int j = 0; j < N; j++) {
                    A[i * N + j] /= sum;
                }
            }
            
            // Compute O_h = A @ V_h
            for (int i = 0; i < N; i++) {               // row of A, i.e. i-th row of A
                for (int k = 0; k < d_h; k++) {           // column of V, i.e. k-th column of V
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        float a_val = A[i * N + j];
                        float v_val = V[b * N * d + j * d + h * d_h + k];
                        sum += a_val * v_val;
                    }
                    output[b * N * d + i * d + h * d_h + k] = sum;
                }
            }
        }
    }
}

bool compare_results(const float* result, const float* expected, size_t n, float tolerance = 1e-4f) {
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(result[i] - expected[i]);
        float maxAbs = std::max(std::abs(result[i]), std::abs(expected[i]));
        float effectiveTol = std::max(tolerance, tolerance * maxAbs);
        if (diff > effectiveTol) {
            return false;
        }
    }
    return true;
}

void test_case(int B, int N, int d, int num_heads, const std::string& name) {
    std::cout << "\n=== Test: " << name << " ===" << std::endl;
    std::cout << "B=" << B << ", N=" << N << ", d=" << d << ", num_heads=" << num_heads << std::endl;
    
    size_t total_size = (size_t)B * N * d;
    
    // Allocate host memory
    float* h_Q = alloc_host(total_size);
    float* h_K = alloc_host(total_size);
    float* h_V = alloc_host(total_size);
    float* h_output_cpu = alloc_host(total_size);
    float* h_output_gpu = alloc_host(total_size);
    
    // Initialize with random data
    rand_init(h_Q, total_size);
    rand_init(h_K, total_size);
    rand_init(h_V, total_size);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_multi_head_attention(h_Q, h_K, h_V, h_output_cpu, B, N, d, num_heads);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    
    // Allocate device memory
    float* d_Q = alloc_device(total_size);
    float* d_K = alloc_device(total_size);
    float* d_V = alloc_device(total_size);
    float* d_output = alloc_device(total_size);
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, total_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // GPU solve
    double gpu_time = benchmark([&]() {
        solve(d_Q, d_K, d_V, d_output, B, N, d, num_heads);
    }, 50);
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare results
    bool correct = compare_results(h_output_gpu, h_output_cpu, total_size);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU time: " << cpu_time << " us" << std::endl;
    std::cout << "GPU time: " << gpu_time << " us (avg of 50 runs)" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    if (!correct) {
        // Find first mismatch
        int mismatches = 0;
        for (size_t i = 0; i < total_size && mismatches < 5; i++) {
            float diff = std::abs(h_output_cpu[i] - h_output_gpu[i]);
            if (diff > 1e-4f) {
                int b = i / (N * d);
                int n = (i % (N * d)) / d;
                int k = i % d;
                std::cout << "  Mismatch at [" << b << "," << n << "," << k << "]: "
                          << "CPU=" << h_output_cpu[i] << ", GPU=" << h_output_gpu[i] 
                          << ", diff=" << diff << std::endl;
                mismatches++;
            }
        }
    }
    
    // Cleanup
    free_host(h_Q);
    free_host(h_K);
    free_host(h_V);
    free_host(h_output_cpu);
    free_host(h_output_gpu);
    free_device(d_Q);
    free_device(d_K);
    free_device(d_V);
    free_device(d_output);
}

void benchmark_test(int B, int N, int d, int num_heads) {
    std::cout << "\n=== Benchmark: B=" << B << ", N=" << N << ", d=" << d 
              << ", heads=" << num_heads << " ===" << std::endl;
    
    size_t total_size = (size_t)B * N * d;
    
    // Allocate and initialize
    float* h_Q = alloc_host(total_size);
    float* h_K = alloc_host(total_size);
    float* h_V = alloc_host(total_size);
    
    rand_init(h_Q, total_size);
    rand_init(h_K, total_size);
    rand_init(h_V, total_size);
    
    float* d_Q = alloc_device(total_size);
    float* d_K = alloc_device(total_size);
    float* d_V = alloc_device(total_size);
    float* d_output = alloc_device(total_size);
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, total_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark
    double gpu_time = benchmark([&]() {
        solve(d_Q, d_K, d_V, d_output, B, N, d, num_heads);
    }, 100);
    
    // Compute theoretical FLOPS
    // Q @ K^T: 2 * B * num_heads * N * N * d_h
    // Softmax: ~3 * B * num_heads * N * N (exp, sum, div)
    // A @ V: 2 * B * num_heads * N * N * d_h
    int d_h = d / num_heads;
    double flops = 4.0 * B * num_heads * N * N * d_h + 3.0 * B * num_heads * N * N;
    double tflops = flops / (gpu_time * 1e-6) / 1e12;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GPU time: " << gpu_time << " us" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;
    
    // Cleanup
    free_host(h_Q);
    free_host(h_K);
    free_host(h_V);
    free_device(d_Q);
    free_device(d_K);
    free_device(d_V);
    free_device(d_output);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "=== Multi-Head Self-Attention Tests ===" << std::endl;
    
    // Small tests for correctness
    test_case(1, 4, 8, 2, "Tiny: 1 batch, 4 seq, 8 dim, 2 heads");
    test_case(1, 8, 16, 4, "Small: 1 batch, 8 seq, 16 dim, 4 heads");
    test_case(2, 16, 32, 4, "Medium: 2 batches, 16 seq, 32 dim, 4 heads");
    test_case(4, 32, 64, 8, "Standard: 4 batches, 32 seq, 64 dim, 8 heads");
    test_case(1, 64, 128, 8, "Large seq: 1 batch, 64 seq, 128 dim, 8 heads");
    
    // Typical transformer configurations
    test_case(1, 128, 256, 8, "Transformer-like: 1 batch, 128 seq, 256 dim, 8 heads");
    test_case(4, 64, 512, 8, "GPT-style: 4 batches, 64 seq, 512 dim, 8 heads");
    
    std::cout << "\n\n=== Performance Benchmarks ===" << std::endl;
    
    // Benchmark with various sizes
    benchmark_test(1, 64, 64, 4);
    benchmark_test(1, 128, 128, 8);
    benchmark_test(1, 256, 256, 8);
    benchmark_test(4, 128, 256, 8);
    benchmark_test(8, 256, 512, 8);
    benchmark_test(16, 128, 768, 12);  // BERT-base like
    
    return 0;
}

