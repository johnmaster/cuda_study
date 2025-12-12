#include "kernels.cuh"
#include <cmath>

// CPU reference implementation for subarray sum
int64_t cpu_subarray_sum(int32_t* input, size_t n, size_t S, size_t E) {
    if (S > E || S >= n) return 0;
    if (E >= n) E = n - 1;
    
    int64_t sum = 0;
    for (size_t i = S; i <= E; i++) {
        sum += input[i];
    }
    return sum;
}

// Verify GPU result against CPU reference
bool verify(int64_t cpu_result, int32_t gpu_result) {
    int32_t expected = static_cast<int32_t>(cpu_result);
    if (expected != gpu_result) {
        std::cerr << "Mismatch: CPU = " << cpu_result 
                  << " (truncated: " << expected << ")"
                  << ", GPU = " << gpu_result << std::endl;
        return false;
    }
    return true;
}

void test_subarray_sum(size_t n, size_t S, size_t E, int epochs = 100) {
    std::cout << "\n=====================" << std::endl;
    std::cout << "Testing Subarray Sum" << std::endl;
    std::cout << "N = " << n << ", S = " << S << ", E = " << E << std::endl;
    std::cout << "Range length = " << (E >= S ? E - S + 1 : 0) << std::endl;
    std::cout << "=====================\n" << std::endl;

    // Allocate host memory
    int32_t* h_input = alloc_host(n);
    int32_t h_result_gpu = 0;
    
    // Allocate device memory
    int32_t* d_input = alloc_device(n);
    int32_t* d_result = alloc_device(1);

    // Initialize input array with random values
    rand_init(h_input, n);
    
    // Print sample of input array
    std::cout << "Input (first 5): [";
    for (size_t i = 0; i < std::min(n, (size_t)5); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << h_input[i];
    }
    if (n > 5) std::cout << ", ...";
    std::cout << "]" << std::endl;
    
    // Print elements in range
    if (S < n && S <= E) {
        size_t actual_E = std::min(E, n - 1);
        std::cout << "Range [" << S << ".." << actual_E << "] (first 5): [";
        for (size_t i = S; i <= std::min(S + 4, actual_E); i++) {
            if (i > S) std::cout << ", ";
            std::cout << h_input[i];
        }
        if (actual_E > S + 4) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int64_t cpu_result = cpu_subarray_sum(h_input, n, S, E);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " us" << std::endl;

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // GPU computation with benchmarking
    double gpu_time = benchmark(
        [&]() {
            solve(d_input, d_result, n, S, E);
        }, epochs
    );
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(&h_result_gpu, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::cout << "GPU result: " << h_result_gpu << std::endl;
    std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpu_time << " us" << std::endl;
    
    // Verify correctness
    bool match = verify(cpu_result, h_result_gpu);
    std::cout << "Result match: " << (match ? "YES" : "NO") << std::endl;

    // Cleanup
    free_host(h_input);
    free_device(d_input);
    free_device(d_result);

    if (!match) {
        std::cerr << "Test FAILED" << std::endl;
        exit(1);
    }
    std::cout << "TEST PASSED" << std::endl;
}

int main() {
    CHECK_CUDA(cudaDeviceReset());

    // Test various sizes and ranges
    test_subarray_sum(8, 0, 7, 100);           // Full array, very small
    test_subarray_sum(8, 2, 5, 100);           // Subrange, very small
    test_subarray_sum(256, 0, 255, 100);       // Full array, single block
    test_subarray_sum(256, 50, 150, 100);      // Subrange, single block
    test_subarray_sum(1024, 100, 900, 100);    // Subrange, multiple blocks
    test_subarray_sum(1 << 14, 0, (1 << 14) - 1, 100);     // 16K elements, full
    test_subarray_sum(1 << 14, 1000, 10000, 100);          // 16K elements, subrange
    test_subarray_sum(1 << 18, 0, (1 << 18) - 1, 100);     // 256K elements, full
    test_subarray_sum(1 << 18, 50000, 200000, 100);        // 256K elements, subrange
    test_subarray_sum(1 << 20, 0, (1 << 20) - 1, 100);     // 1M elements, full
    test_subarray_sum(1 << 20, 100000, 900000, 100);       // 1M elements, subrange
    
    // Edge cases
    test_subarray_sum(100, 0, 0, 100);         // Single element at start
    test_subarray_sum(100, 99, 99, 100);       // Single element at end
    test_subarray_sum(100, 50, 50, 100);       // Single element in middle
    
    std::cout << "\n============ All tests passed ===========\n" << std::endl;
    return 0;
}
