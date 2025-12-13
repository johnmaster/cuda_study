#include "kernels.cuh"

// CPU reference implementation
int32_t cpu_max_subarray_sum(int32_t* data, size_t N, size_t window_size) {
    if (window_size > N || window_size == 0) {
        return INT_MIN;
    }
    
    // Calculate first window sum
    int32_t window_sum = 0;
    for (size_t i = 0; i < window_size; i++) {
        window_sum += data[i];
    }
    
    int32_t max_sum = window_sum;
    
    // Slide window and track maximum
    for (size_t i = window_size; i < N; i++) {
        window_sum = window_sum - data[i - window_size] + data[i];
        max_sum = std::max(max_sum, window_sum);
    }
    
    return max_sum;
}

void test_case(const std::vector<int32_t>& input, size_t window_size, 
               int32_t expected, const std::string& name) {
    size_t N = input.size();
    
    // Allocate device memory
    int32_t* d_input = alloc_device(N);
    int32_t* d_output = alloc_device(1);
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // Run solve
    solve(d_input, d_output, N, window_size);
    
    // Get result
    int32_t result;
    CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    // Verify
    bool passed = (result == expected);
    std::cout << name << ": ";
    if (passed) {
        std::cout << "PASSED (output = " << result << ")" << std::endl;
    } else {
        std::cout << "FAILED (expected = " << expected << ", got = " << result << ")" << std::endl;
    }
    
    // Cleanup
    free_device(d_input);
    free_device(d_output);
}

void benchmark_test(size_t N, size_t window_size) {
    std::cout << "\n=== Benchmark: N = " << N << ", window_size = " << window_size << " ===" << std::endl;
    
    // Allocate and initialize host data
    int32_t* h_data = alloc_host(N);
    rand_init(h_data, N);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int32_t cpu_result = cpu_max_subarray_sum(h_data, N, window_size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    
    // Allocate device memory
    int32_t* d_input = alloc_device(N);
    int32_t* d_output = alloc_device(1);
    CHECK_CUDA(cudaMemcpy(d_input, h_data, N * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // GPU benchmark
    double gpu_time = benchmark([&]() {
        solve(d_input, d_output, N, window_size);
    }, 100);
    
    // Verify result
    int32_t gpu_result;
    CHECK_CUDA(cudaMemcpy(&gpu_result, d_output, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    bool correct = (cpu_result == gpu_result);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU time: " << cpu_time << " us" << std::endl;
    std::cout << "GPU time: " << gpu_time << " us (avg of 100 runs)" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") 
              << " (CPU: " << cpu_result << ", GPU: " << gpu_result << ")" << std::endl;
    
    // Cleanup
    free_device(d_input);
    free_device(d_output);
    free_host(h_data);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "=== Max Subarray Sum Tests ===" << std::endl;
    
    // Test Example 1: input = [1, 2, 4, 2, 3], window_size = 2, output = 6
    test_case({1, 2, 4, 2, 3}, 2, 6, "Example 1");
    
    // Test Example 2: input = [-1, -4, -2, 1], window_size = 3, output = -5
    test_case({-1, -4, -2, 1}, 3, -5, "Example 2");
    
    // Additional test cases
    test_case({1, 2, 3, 4, 5}, 1, 5, "Single element window");
    test_case({1, 2, 3, 4, 5}, 5, 15, "Full array window");
    test_case({-5, -3, -1, -2, -4}, 2, -3, "All negative");
    test_case({10}, 1, 10, "Single element array");
    test_case({5, -2, 8, -1, 4}, 3, 11, "Mixed values");
    
    // Benchmark tests
    benchmark_test(1 << 16, 64);      // 64K elements, window = 64
    benchmark_test(1 << 20, 128);     // 1M elements, window = 128
    benchmark_test(1 << 20, 1024);    // 1M elements, window = 1024
    benchmark_test(1 << 22, 256);     // 4M elements, window = 256
    benchmark_test(1 << 24, 64);      // 16M elements, window = 64
    benchmark_test(1 << 24, 256);     // 16M elements, window = 256
    benchmark_test(1 << 24, 1024);    // 16M elements, window = 1024
    
    cleanup_workspace();
    return 0;
}
