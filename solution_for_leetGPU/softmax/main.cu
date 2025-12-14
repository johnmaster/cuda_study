#include "kernels.cuh"

// CPU reference implementation with max trick for numerical stability
void cpu_softmax(const float* input, float* output, size_t N) {
    if (N == 0) return;
    
    // Find maximum
    float maxVal = input[0];
    for (size_t i = 1; i < N; i++) {
        maxVal = std::max(maxVal, input[i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    
    // Normalize
    for (size_t i = 0; i < N; i++) {
        output[i] /= sum;
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
               const std::string& name, float tolerance = 1e-5f) {
    size_t N = input.size();
    
    // Allocate device memory
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run solve
    solve(d_input, d_output, N);
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

void test_case_auto(const std::vector<float>& input, const std::string& name) {
    size_t N = input.size();
    std::vector<float> expected(N);
    cpu_softmax(input.data(), expected.data(), N);
    test_case(input, expected, name);
}

void benchmark_test(size_t N) {
    std::cout << "\n=== Benchmark: N = " << N << " ===" << std::endl;
    
    // Allocate and initialize host data
    float* h_input = alloc_host(N);
    rand_init(h_input, N);
    
    float* h_output_cpu = alloc_host(N);
    float* h_output_gpu = alloc_host(N);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_softmax(h_input, h_output_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    
    // Allocate device memory
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // GPU benchmark
    double gpu_time = benchmark([&]() {
        solve(d_input, d_output, N);
    }, 100);
    
    // Verify result
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = compare_results(h_output_gpu, h_output_cpu, N, 1e-4f);
    
    // Check sum is 1.0
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        sum += h_output_gpu[i];
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU time: " << cpu_time << " us" << std::endl;
    std::cout << "GPU time: " << gpu_time << " us (avg of 100 runs)" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Sum of probabilities: " << std::setprecision(6) << sum << std::endl;
    
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
    
    std::cout << "=== Softmax Tests ===" << std::endl;
    
    // Test Example 1: Simple case
    // softmax([1, 2, 3]) = [0.0900, 0.2447, 0.6652]
    test_case({1.0f, 2.0f, 3.0f}, 
              {0.0900306f, 0.244728f, 0.665241f}, 
              "Simple case [1,2,3]");
    
    // Test Example 2: All same values -> uniform distribution
    test_case({1.0f, 1.0f, 1.0f, 1.0f}, 
              {0.25f, 0.25f, 0.25f, 0.25f}, 
              "Uniform [1,1,1,1]");
    
    // Test Example 3: Single element -> 1.0
    test_case({5.0f}, {1.0f}, "Single element");
    
    // Test Example 4: Two elements
    test_case({0.0f, 0.0f}, {0.5f, 0.5f}, "Two equal elements");
    
    // Test Example 5: Large values (tests max trick for overflow prevention)
    test_case({1000.0f, 1001.0f, 1002.0f}, 
              {0.0900306f, 0.244728f, 0.665241f}, 
              "Large values [1000,1001,1002]");
    
    // Test Example 6: Negative values
    test_case({-1.0f, -2.0f, -3.0f}, 
              {0.665241f, 0.244728f, 0.0900306f}, 
              "Negative values [-1,-2,-3]");
    
    // Test Example 7: Mixed positive and negative
    test_case_auto({-2.0f, 0.0f, 2.0f, 4.0f}, "Mixed values [-2,0,2,4]");
    
    // Test Example 8: Zero input
    test_case({0.0f, 0.0f, 0.0f}, 
              {0.333333f, 0.333333f, 0.333333f}, 
              "All zeros");
    
    // Test Example 9: Larger array
    std::vector<float> large_input(100);
    for (int i = 0; i < 100; i++) {
        large_input[i] = static_cast<float>(i) * 0.1f;
    }
    test_case_auto(large_input, "100 elements");
    
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
