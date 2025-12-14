#include "kernels.cuh"

// CPU reference implementation
void cpu_rms_norm(const float* input, float* output, float gamma, float beta, size_t N, float eps) {
    if (N == 0) return;
    
    // Compute sum of squares
    float sumSq = 0.0f;
    for (size_t i = 0; i < N; i++) {
        sumSq += input[i] * input[i];
    }
    
    // Compute RMS = sqrt(mean(x^2) + eps)
    float meanSq = sumSq / static_cast<float>(N);
    float rms = std::sqrt(meanSq + eps);
    
    // Apply normalization: output = gamma * (x / rms) + beta
    for (size_t i = 0; i < N; i++) {
        output[i] = gamma * (input[i] / rms) + beta;
    }
}

bool compare_results(const float* result, const float* expected, size_t N, float tolerance = 1e-5f) {
    for (size_t i = 0; i < N; i++) {
        float diff = std::abs(result[i] - expected[i]);
        float maxAbs = std::max(std::abs(result[i]), std::abs(expected[i]));
        // Use relative tolerance for larger values, absolute for small
        float effectiveTol = std::max(tolerance, tolerance * maxAbs);
        if (diff > effectiveTol) {
            return false;
        }
    }
    return true;
}

void test_case(const std::vector<float>& input, const std::vector<float>& expected,
               float gamma, float beta, float eps, const std::string& name, float tolerance = 1e-5f) {
    size_t N = input.size();
    
    // Allocate device memory
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run solve
    solve(d_input, d_output, gamma, beta, N, eps);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Get result
    std::vector<float> result(N);
    CHECK_CUDA(cudaMemcpy(result.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    bool passed = compare_results(result.data(), expected.data(), N, tolerance);
    std::cout << name << ": ";
    if (passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "  Expected: [";
        for (size_t i = 0; i < std::min(N, (size_t)10); i++) {
            std::cout << std::fixed << std::setprecision(6) << expected[i];
            if (i < std::min(N, (size_t)10) - 1) std::cout << ", ";
        }
        if (N > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < std::min(N, (size_t)10); i++) {
            std::cout << std::fixed << std::setprecision(6) << result[i];
            if (i < std::min(N, (size_t)10) - 1) std::cout << ", ";
        }
        if (N > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    
    // Cleanup
    free_device(d_input);
    free_device(d_output);
}

void test_case_auto(const std::vector<float>& input, float gamma, float beta, float eps,
                    const std::string& name) {
    size_t N = input.size();
    std::vector<float> expected(N);
    cpu_rms_norm(input.data(), expected.data(), gamma, beta, N, eps);
    test_case(input, expected, gamma, beta, eps, name);
}

void benchmark_test(size_t N) {
    std::cout << "\n=== Benchmark: N = " << N << " ===" << std::endl;
    
    float gamma = 1.0f;
    float beta = 0.0f;
    float eps = 1e-5f;
    
    // Allocate and initialize host data
    float* h_input = alloc_host(N);
    rand_init(h_input, N);
    
    float* h_output_cpu = alloc_host(N);
    float* h_output_gpu = alloc_host(N);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_rms_norm(h_input, h_output_cpu, gamma, beta, N, eps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    
    // Allocate device memory
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // GPU benchmark
    double gpu_time = benchmark([&]() {
        solve(d_input, d_output, gamma, beta, N, eps);
    }, 100);
    
    // Verify result
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Use larger tolerance for bigger arrays due to accumulated FP precision differences
    float tolerance = 1e-4f;
    if (N > 1000000) tolerance = 5e-2f;  // Very large arrays have more FP accumulation error
    else if (N > 100000) tolerance = 1e-3f;
    bool correct = compare_results(h_output_gpu, h_output_cpu, N, tolerance);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU time: " << cpu_time << " us" << std::endl;
    std::cout << "GPU time: " << gpu_time << " us (avg of 100 runs)" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    if (!correct) {
        // Print first few mismatches
        int mismatches = 0;
        for (size_t i = 0; i < N && mismatches < 5; i++) {
            float diff = std::abs(h_output_cpu[i] - h_output_gpu[i]);
            if (diff > 1e-4f) {
                std::cout << "  Mismatch at " << i << ": CPU=" << h_output_cpu[i] 
                          << ", GPU=" << h_output_gpu[i] << ", diff=" << diff << std::endl;
                mismatches++;
            }
        }
    }
    
    // Cleanup
    free_device(d_input);
    free_device(d_output);
    free_host(h_input);
    free_host(h_output_cpu);
    free_host(h_output_gpu);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "=== RMS Normalization Tests ===" << std::endl;
    
    // Test Example 1: From problem statement
    // input = [1.0, 2.0, 3.0, 4.0], gamma = 1.0, beta = 0.0, eps = 1e-5
    // RMS = sqrt((1 + 4 + 9 + 16)/4 + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.7386128
    // output = [0.36514813, 0.73029625, 1.0954444, 1.4605925]
    test_case({1.0f, 2.0f, 3.0f, 4.0f}, 
              {0.36514813f, 0.73029625f, 1.0954444f, 1.4605925f}, 
              1.0f, 0.0f, 1e-5f,
              "Example from problem [1,2,3,4]");
    
    // Test Example 2: All same values
    test_case_auto({2.0f, 2.0f, 2.0f, 2.0f}, 1.0f, 0.0f, 1e-5f,
                   "Uniform [2,2,2,2]");
    
    // Test Example 3: Single element
    test_case_auto({5.0f}, 1.0f, 0.0f, 1e-5f, "Single element");
    
    // Test Example 4: With scale and shift
    test_case_auto({1.0f, 2.0f, 3.0f, 4.0f}, 2.0f, 0.5f, 1e-5f,
                   "With gamma=2.0, beta=0.5");
    
    // Test Example 5: Negative values
    test_case_auto({-1.0f, -2.0f, -3.0f, -4.0f}, 1.0f, 0.0f, 1e-5f,
                   "Negative values [-1,-2,-3,-4]");
    
    // Test Example 6: Mixed positive and negative
    test_case_auto({-2.0f, 0.0f, 2.0f, 4.0f}, 1.0f, 0.0f, 1e-5f,
                   "Mixed values [-2,0,2,4]");
    
    // Test Example 7: Larger array
    std::vector<float> large_input(100);
    for (int i = 0; i < 100; i++) {
        large_input[i] = static_cast<float>(i) * 0.1f;
    }
    test_case_auto(large_input, 1.0f, 0.0f, 1e-5f, "100 elements");
    
    // Test Example 8: Very small values (test eps)
    test_case_auto({1e-6f, 1e-6f, 1e-6f}, 1.0f, 0.0f, 1e-5f,
                   "Very small values");
    
    // Benchmark tests
    benchmark_test(1 << 10);    // 1K elements
    benchmark_test(1 << 14);    // 16K elements
    benchmark_test(1 << 16);    // 64K elements
    benchmark_test(1 << 18);    // 256K elements
    benchmark_test(1 << 20);    // 1M elements
    benchmark_test(1 << 22);    // 4M elements
    benchmark_test(1 << 24);    // 16M elements
    
    return 0;
}
