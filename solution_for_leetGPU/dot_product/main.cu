#include "kernels.cuh"
#include <cmath>

// CPU reference implementation for dot product
float cpu_dot_product(float* a, float* b, size_t n) {
    double sum = 0.0;  // Use double for better precision in accumulation
    for (size_t i = 0; i < n; i++) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}

// Verify GPU result against CPU reference
bool verify(float cpu_result, float gpu_result, float tolerance = 1e-3f) {
    float diff = std::abs(cpu_result - gpu_result);
    float max_val = std::max(std::abs(cpu_result), std::abs(gpu_result));
    float rel_error = (max_val > 1e-6f) ? diff / max_val : diff;
    
    if (rel_error > tolerance && diff > tolerance) {
        std::cerr << "Mismatch: CPU = " << cpu_result 
                  << ", GPU = " << gpu_result 
                  << ", diff = " << diff 
                  << ", rel_error = " << rel_error << std::endl;
        return false;
    }
    return true;
}

void test_dot_product(size_t n, int epochs = 100) {
    std::cout << "\n=====================" << std::endl;
    std::cout << "Testing Dot Product (N = " << n << ")" << std::endl;
    std::cout << "=====================\n" << std::endl;

    // Allocate host memory
    float* h_a = alloc_host(n);
    float* h_b = alloc_host(n);
    float h_result_gpu = 0.0f;
    
    // Allocate device memory
    float* d_a = alloc_device(n);
    float* d_b = alloc_device(n);
    float* d_result = alloc_device(1);

    // Initialize input vectors with random values
    rand_init(h_a, n);
    rand_init(h_b, n);
    
    // Print sample of input vectors
    std::cout << "Vector A (first 5): [";
    for (size_t i = 0; i < std::min(n, (size_t)5); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << h_a[i];
    }
    if (n > 5) std::cout << ", ...";
    std::cout << "]" << std::endl;
    
    std::cout << "Vector B (first 5): [";
    for (size_t i = 0; i < std::min(n, (size_t)5); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << h_b[i];
    }
    if (n > 5) std::cout << ", ...";
    std::cout << "]" << std::endl;

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = cpu_dot_product(h_a, h_b, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    std::cout << "CPU result: " << std::fixed << std::setprecision(6) << cpu_result << std::endl;
    std::cout << "CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " us" << std::endl;

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // GPU computation with benchmarking
    double gpu_time = benchmark(
        [&]() {
            solve(d_a, d_b, d_result, n);
        }, epochs
    );
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(&h_result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "GPU result: " << std::fixed << std::setprecision(6) << h_result_gpu << std::endl;
    std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpu_time << " us" << std::endl;
    
    // Verify correctness
    bool match = verify(cpu_result, h_result_gpu);
    std::cout << "Result match: " << (match ? "YES" : "NO") << std::endl;

    // Cleanup
    free_host(h_a);
    free_host(h_b);
    free_device(d_a);
    free_device(d_b);
    free_device(d_result);

    if (!match) {
        std::cerr << "Test FAILED" << std::endl;
        exit(1);
    }
    std::cout << "TEST PASSED" << std::endl;
}

int main() {
    CHECK_CUDA(cudaDeviceReset());

    // Test various sizes - powers of 2
    test_dot_product(8, 100);           // Very small
    test_dot_product(256, 100);         // Single block
    test_dot_product(512, 100);         // Multiple blocks (standard kernel)
    test_dot_product(1024, 100);        // Threshold for vectorized kernel
    test_dot_product(1 << 14, 100);     // 16K elements
    test_dot_product(1 << 18, 100);     // 256K elements
    test_dot_product(1 << 20, 100);     // 1M elements
    test_dot_product(1 << 24, 100);     // 16M elements
    
    // Test odd sizes and non-aligned lengths
    test_dot_product(1, 100);           // Single element
    test_dot_product(3, 100);           // Very small odd
    test_dot_product(7, 100);           // Small odd
    test_dot_product(33, 100);          // Odd > warp size
    test_dot_product(127, 100);         // Odd near block boundary
    test_dot_product(257, 100);         // Odd > single block
    test_dot_product(1023, 100);        // Odd, just below vectorized threshold
    test_dot_product(1025, 100);        // Odd, just above threshold (uses standard kernel)
    test_dot_product(10007, 100);       // Large prime number
    test_dot_product(100003, 100);      // Larger prime
    test_dot_product((1 << 20) + 1, 100);  // 1M + 1 (odd large)
    test_dot_product((1 << 20) + 3, 100);  // 1M + 3 (odd large, not 4-aligned)
    
    std::cout << "\n============ All tests passed ===========\n" << std::endl;
    return 0;
}
