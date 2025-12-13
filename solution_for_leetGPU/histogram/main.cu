#include "kernels.cuh"

// CPU reference implementation
void cpu_histogram(const int32_t* input, int32_t* histogram, size_t N, int32_t num_bins) {
    // Zero out histogram
    for (int32_t i = 0; i < num_bins; i++) {
        histogram[i] = 0;
    }
    
    // Count occurrences
    for (size_t i = 0; i < N; i++) {
        int32_t val = input[i];
        if (val >= 0 && val < num_bins) {
            histogram[val]++;
        }
    }
}

bool compare_histograms(const int32_t* h1, const int32_t* h2, int32_t num_bins) {
    for (int32_t i = 0; i < num_bins; i++) {
        if (h1[i] != h2[i]) {
            return false;
        }
    }
    return true;
}

void test_case(const std::vector<int32_t>& input, int32_t num_bins,
               const std::vector<int32_t>& expected, const std::string& name) {
    size_t N = input.size();
    
    // Allocate device memory
    int32_t* d_input = alloc_device(N);
    int32_t* d_histogram = alloc_device(num_bins);
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // Run solve
    solve(d_input, d_histogram, N, num_bins);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Get result
    std::vector<int32_t> result(num_bins);
    CHECK_CUDA(cudaMemcpy(result.data(), d_histogram, num_bins * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    // Verify
    bool passed = (result == expected);
    std::cout << name << ": ";
    if (passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "  Expected: [";
        for (size_t i = 0; i < expected.size(); i++) {
            std::cout << expected[i] << (i < expected.size()-1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i] << (i < result.size()-1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }
    
    // Cleanup
    free_device(d_input);
    free_device(d_histogram);
}

void benchmark_test(size_t N, int32_t num_bins) {
    std::cout << "\n=== Benchmark: N = " << N << ", num_bins = " << num_bins << " ===" << std::endl;
    
    // Allocate and initialize host data
    int32_t* h_input = alloc_host(N);
    rand_init(h_input, N, num_bins);
    
    int32_t* h_histogram_cpu = alloc_host(num_bins);
    int32_t* h_histogram_gpu = alloc_host(num_bins);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_histogram(h_input, h_histogram_cpu, N, num_bins);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    
    // Allocate device memory
    int32_t* d_input = alloc_device(N);
    int32_t* d_histogram = alloc_device(num_bins);
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // GPU benchmark
    double gpu_time = benchmark([&]() {
        solve(d_input, d_histogram, N, num_bins);
    }, 100);
    
    // Verify result
    CHECK_CUDA(cudaMemcpy(h_histogram_gpu, d_histogram, num_bins * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    bool correct = compare_histograms(h_histogram_cpu, h_histogram_gpu, num_bins);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU time: " << cpu_time << " us" << std::endl;
    std::cout << "GPU time: " << gpu_time << " us (avg of 100 runs)" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    if (!correct) {
        // Print first few mismatches
        int mismatches = 0;
        for (int32_t i = 0; i < num_bins && mismatches < 5; i++) {
            if (h_histogram_cpu[i] != h_histogram_gpu[i]) {
                std::cout << "  Mismatch at bin " << i << ": CPU=" << h_histogram_cpu[i] 
                          << ", GPU=" << h_histogram_gpu[i] << std::endl;
                mismatches++;
            }
        }
    }
    
    // Cleanup
    free_device(d_input);
    free_device(d_histogram);
    free_host(h_input);
    free_host(h_histogram_cpu);
    free_host(h_histogram_gpu);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "=== Histogram Tests ===" << std::endl;
    
    // Test Example 1: Simple case
    test_case({0, 1, 1, 2, 3, 3, 3}, 4, {1, 2, 1, 3}, "Simple histogram");
    
    // Test Example 2: All same value
    test_case({2, 2, 2, 2, 2}, 4, {0, 0, 5, 0}, "All same value");
    
    // Test Example 3: Sequential values
    test_case({0, 1, 2, 3, 4}, 5, {1, 1, 1, 1, 1}, "Sequential values");
    
    // Test Example 4: With out-of-range values (should be ignored)
    test_case({-1, 0, 1, 5, 2, 10}, 4, {1, 1, 1, 0}, "With out-of-range values");
    
    // Test Example 5: Empty valid values
    test_case({-5, -3, 10, 20}, 4, {0, 0, 0, 0}, "All out-of-range");
    
    // Test Example 6: Single element
    test_case({3}, 5, {0, 0, 0, 1, 0}, "Single element");
    
    // Test Example 7: All zeros
    test_case({0, 0, 0, 0}, 3, {4, 0, 0}, "All zeros");
    
    // Benchmark tests
    benchmark_test(1 << 16, 256);      // 64K elements, 256 bins
    benchmark_test(1 << 20, 256);      // 1M elements, 256 bins
    benchmark_test(1 << 20, 1024);     // 1M elements, 1024 bins
    benchmark_test(1 << 22, 256);      // 4M elements, 256 bins
    benchmark_test(1 << 24, 256);      // 16M elements, 256 bins
    benchmark_test(1 << 24, 4096);     // 16M elements, 4096 bins
    
    return 0;
}
