#include "kernels.cuh"
#include <cmath>

// CPU reference implementation for dot product using FP32 accumulation
// Takes half precision input but accumulates in double for reference precision
float cpu_dot_product(half* a, half* b, size_t n) {
    double sum = 0.0;  // Use double for best precision in reference
    for (size_t i = 0; i < n; i++) {
        // Convert half to float for computation
        float a_val = __half2float(a[i]);
        float b_val = __half2float(b[i]);
        sum += static_cast<double>(a_val) * static_cast<double>(b_val);
    }
    return static_cast<float>(sum);
}

// Verify GPU result against CPU reference
// Note: FP16 has lower precision, so we use a larger tolerance
// Also handles overflow to inf for large results (expected for FP16)
bool verify(float cpu_result, float gpu_result, float tolerance = 1e-2f) {
    // Handle infinity cases - FP16 max is ~65504
    const float FP16_MAX = 65504.0f;
    
    // If GPU result is inf, check if CPU result also exceeds FP16 range
    if (std::isinf(gpu_result)) {
        if (std::abs(cpu_result) > FP16_MAX) {
            // Expected overflow - both should be inf (same sign)
            std::cout << "Note: Result exceeds FP16 range (|" << cpu_result 
                      << "| > 65504), overflow to inf is expected" << std::endl;
            return (gpu_result > 0) == (cpu_result > 0);  // Same sign check
        }
        std::cerr << "Unexpected overflow: CPU = " << cpu_result 
                  << ", GPU = " << gpu_result << std::endl;
        return false;
    }
    
    // Handle NaN
    if (std::isnan(cpu_result) || std::isnan(gpu_result)) {
        std::cerr << "NaN detected: CPU = " << cpu_result 
                  << ", GPU = " << gpu_result << std::endl;
        return std::isnan(cpu_result) && std::isnan(gpu_result);
    }
    
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
    std::cout << "Testing FP16 Dot Product (N = " << n << ")" << std::endl;
    std::cout << "=====================\n" << std::endl;

    // Allocate host memory (half precision)
    half* h_a = alloc_host(n);
    half* h_b = alloc_host(n);
    half h_result_gpu;
    
    // Allocate device memory (half precision)
    half* d_a = alloc_device(n);
    half* d_b = alloc_device(n);
    half* d_result = alloc_device(1);

    // Initialize input vectors with random values
    rand_init(h_a, n);
    rand_init(h_b, n);
    
    // Print sample of input vectors
    std::cout << "Vector A (first 5): [";
    for (size_t i = 0; i < std::min(n, (size_t)5); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_a[i]);
    }
    if (n > 5) std::cout << ", ...";
    std::cout << "]" << std::endl;
    
    std::cout << "Vector B (first 5): [";
    for (size_t i = 0; i < std::min(n, (size_t)5); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_b[i]);
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
    CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(half), cudaMemcpyHostToDevice));
    
    // GPU computation with benchmarking
    double gpu_time = benchmark(
        [&]() {
            solve(d_a, d_b, d_result, n);
        }, epochs
    );
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(&h_result_gpu, d_result, sizeof(half), cudaMemcpyDeviceToHost));
    
    // Convert GPU result from half to float for comparison
    float gpu_result = __half2float(h_result_gpu);

    std::cout << "GPU result: " << std::fixed << std::setprecision(6) << gpu_result << std::endl;
    std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpu_time << " us" << std::endl;
    
    // Verify correctness (using larger tolerance for FP16)
    bool match = verify(cpu_result, gpu_result);
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

    // Test various sizes
    test_dot_product(8, 100);           // Very small
    test_dot_product(256, 100);         // Single block
    test_dot_product(512, 100);         // Multiple blocks (standard kernel)
    test_dot_product(1024, 100);        // Threshold for vectorized kernel
    test_dot_product(1 << 14, 100);     // 16K elements
    test_dot_product(1 << 18, 100);     // 256K elements
    test_dot_product(1 << 20, 100);     // 1M elements
    test_dot_product(1 << 24, 100);     // 16M elements
    
    std::cout << "\n============ All tests passed ===========\n" << std::endl;
    return 0;
}
