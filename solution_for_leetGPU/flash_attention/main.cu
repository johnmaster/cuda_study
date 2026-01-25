/**
 * Flash Attention Test and Benchmark
 */

#include "kernels.cuh"

// CPU reference implementation
void cpu_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);
    
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            // Compute attention scores for row i
            std::vector<float> scores(N);
            float max_score = -INFINITY;
            
            for (int j = 0; j < N; j++) {
                float score = 0.0f;
                for (int k = 0; k < d; k++) {
                    score += Q[b * N * d + i * d + k] * K[b * N * d + j * d + k];
                }
                score *= scale;
                scores[j] = score;
                max_score = std::max(max_score, score);
            }
            
            // Softmax
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            for (int j = 0; j < N; j++) {
                scores[j] /= sum;
            }
            
            // Output
            for (int k = 0; k < d; k++) {
                float o = 0.0f;
                for (int j = 0; j < N; j++) {
                    o += scores[j] * V[b * N * d + j * d + k];
                }
                O[b * N * d + i * d + k] = o;
            }
        }
    }
}

bool compare_results(const float* a, const float* b, size_t n, float tol = 1e-3f) {
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    if (max_diff > tol) {
        std::cout << "Max diff: " << max_diff << " at index " << max_diff_idx 
                  << " (expected: " << b[max_diff_idx] << ", got: " << a[max_diff_idx] << ")" << std::endl;
        return false;
    }
    return true;
}

void test_correctness(int B, int N, int d) {
    std::cout << "\n=== Correctness Test: B=" << B << ", N=" << N << ", d=" << d << " ===" << std::endl;
    
    size_t size = B * N * d;
    
    // Allocate host memory
    float* h_Q = alloc_host(size);
    float* h_K = alloc_host(size);
    float* h_V = alloc_host(size);
    float* h_O_cpu = alloc_host(size);
    float* h_O_flash = alloc_host(size);
    float* h_O_std = alloc_host(size);
    
    // Initialize
    rand_init(h_Q, size);
    rand_init(h_K, size);
    rand_init(h_V, size);
    
    // CPU reference
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, B, N, d);
    
    // Allocate device memory
    float* d_Q = alloc_device(size);
    float* d_K = alloc_device(size);
    float* d_V = alloc_device(size);
    float* d_O = alloc_device(size);
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test Flash Attention
    flash_attention_forward(d_Q, d_K, d_V, d_O, B, N, d);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_flash, d_O, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool flash_correct = compare_results(h_O_flash, h_O_cpu, size);
    std::cout << "Flash Attention: " << (flash_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Test Standard Attention
    standard_attention(d_Q, d_K, d_V, d_O, B, N, d);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_std, d_O, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool std_correct = compare_results(h_O_std, h_O_cpu, size);
    std::cout << "Standard Attention: " << (std_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    free_host(h_Q);
    free_host(h_K);
    free_host(h_V);
    free_host(h_O_cpu);
    free_host(h_O_flash);
    free_host(h_O_std);
    free_device(d_Q);
    free_device(d_K);
    free_device(d_V);
    free_device(d_O);
}

void benchmark_attention(int B, int N, int d) {
    std::cout << "\n=== Benchmark: B=" << B << ", N=" << N << ", d=" << d << " ===" << std::endl;
    
    size_t size = B * N * d;
    
    // Allocate
    float* h_Q = alloc_host(size);
    float* h_K = alloc_host(size);
    float* h_V = alloc_host(size);
    
    rand_init(h_Q, size);
    rand_init(h_K, size);
    rand_init(h_V, size);
    
    float* d_Q = alloc_device(size);
    float* d_K = alloc_device(size);
    float* d_V = alloc_device(size);
    float* d_O = alloc_device(size);
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark Flash Attention
    double flash_time = benchmark([&]() {
        flash_attention_forward(d_Q, d_K, d_V, d_O, B, N, d);
    }, 50);
    
    // Benchmark Standard Attention
    double std_time = benchmark([&]() {
        standard_attention(d_Q, d_K, d_V, d_O, B, N, d);
    }, 50);
    
    // Memory usage comparison
    size_t flash_mem = B * N * d * sizeof(float) * 4;  // Q, K, V, O only
    size_t std_mem = flash_mem + B * N * N * sizeof(float);  // + S matrix
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Flash Attention:    " << flash_time << " us" << std::endl;
    std::cout << "Standard Attention: " << std_time << " us" << std::endl;
    std::cout << "Speedup: " << std_time / flash_time << "x" << std::endl;
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "  Flash:    " << flash_mem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Standard: " << std_mem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Savings:  " << (std_mem - flash_mem) / 1024.0 / 1024.0 << " MB ("
              << 100.0 * (std_mem - flash_mem) / std_mem << "%)" << std::endl;
    
    // Cleanup
    free_host(h_Q);
    free_host(h_K);
    free_host(h_V);
    free_device(d_Q);
    free_device(d_K);
    free_device(d_V);
    free_device(d_O);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "============================================" << std::endl;
    std::cout << "        Flash Attention Implementation      " << std::endl;
    std::cout << "============================================" << std::endl;
    
    std::cout << "\n--- 算法核心思想 ---" << std::endl;
    std::cout << "1. Tiling: 分块加载 Q, K, V 到 shared memory" << std::endl;
    std::cout << "2. Online Softmax: 一次遍历计算 max 和 sum" << std::endl;
    std::cout << "3. 不存储 N×N 的注意力矩阵 S" << std::endl;
    std::cout << "\n标准 Attention: O(N²) 内存" << std::endl;
    std::cout << "Flash Attention: O(N) 内存" << std::endl;
    
    // Correctness tests
    test_correctness(1, 32, 32);
    test_correctness(1, 64, 64);
    test_correctness(2, 64, 32);
    test_correctness(1, 128, 64);
    
    // Benchmarks
    benchmark_attention(1, 64, 64);
    benchmark_attention(1, 128, 64);
    benchmark_attention(1, 256, 64);
    benchmark_attention(1, 512, 64);
    benchmark_attention(4, 256, 64);
    benchmark_attention(8, 128, 64);
    
    // Large sequence length to show memory savings
    std::cout << "\n=== Large Sequence Test (Memory Savings) ===" << std::endl;
    benchmark_attention(1, 1024, 64);
    benchmark_attention(1, 2048, 64);
    
    return 0;
}

