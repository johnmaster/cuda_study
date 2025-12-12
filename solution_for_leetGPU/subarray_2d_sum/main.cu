#include "kernels.cuh"
#include <cmath>

// CPU reference implementation for 2D subarray sum
int64_t cpu_subarray_sum_2d(int32_t* input, size_t N, size_t M,
                            size_t S_ROW, size_t E_ROW, size_t S_COL, size_t E_COL) {
    if (S_ROW > E_ROW || S_COL > E_COL || S_ROW >= N || S_COL >= M) return 0;
    if (E_ROW >= N) E_ROW = N - 1;
    if (E_COL >= M) E_COL = M - 1;
    
    int64_t sum = 0;
    for (size_t r = S_ROW; r <= E_ROW; r++) {
        for (size_t c = S_COL; c <= E_COL; c++) {
            sum += input[r * M + c];
        }
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

void test_subarray_sum_2d(size_t N, size_t M, 
                          size_t S_ROW, size_t E_ROW, 
                          size_t S_COL, size_t E_COL, 
                          int epochs = 100) {
    std::cout << "\n=============================" << std::endl;
    std::cout << "Testing 2D Subarray Sum" << std::endl;
    std::cout << "Array size: " << N << " x " << M << std::endl;
    std::cout << "Row range: [" << S_ROW << ", " << E_ROW << "]" << std::endl;
    std::cout << "Col range: [" << S_COL << ", " << E_COL << "]" << std::endl;
    
    size_t actual_e_row = std::min(E_ROW, N - 1);
    size_t actual_e_col = std::min(E_COL, M - 1);
    size_t num_rows = (S_ROW <= actual_e_row) ? actual_e_row - S_ROW + 1 : 0;
    size_t num_cols = (S_COL <= actual_e_col) ? actual_e_col - S_COL + 1 : 0;
    std::cout << "Subarray size: " << num_rows << " x " << num_cols 
              << " = " << num_rows * num_cols << " elements" << std::endl;
    std::cout << "=============================\n" << std::endl;

    size_t total_size = N * M;
    
    // Allocate host memory
    int32_t* h_input = alloc_host(total_size);
    int32_t h_result_gpu = 0;
    
    // Allocate device memory
    int32_t* d_input = alloc_device(total_size);
    int32_t* d_result = alloc_device(1);

    // Initialize input array with random values
    rand_init(h_input, total_size);
    
    // Print sample of input array (top-left corner)
    std::cout << "Input (top-left 3x3):" << std::endl;
    for (size_t r = 0; r < std::min(N, (size_t)3); r++) {
        std::cout << "  [";
        for (size_t c = 0; c < std::min(M, (size_t)3); c++) {
            if (c > 0) std::cout << ", ";
            std::cout << std::setw(5) << h_input[r * M + c];
        }
        if (M > 3) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (N > 3) std::cout << "  ..." << std::endl;
    
    // Print sample of subarray
    if (S_ROW < N && S_COL < M && S_ROW <= E_ROW && S_COL <= E_COL) {
        std::cout << "Subarray (top-left 3x3):" << std::endl;
        for (size_t r = S_ROW; r <= std::min(S_ROW + 2, actual_e_row); r++) {
            std::cout << "  [";
            for (size_t c = S_COL; c <= std::min(S_COL + 2, actual_e_col); c++) {
                if (c > S_COL) std::cout << ", ";
                std::cout << std::setw(5) << h_input[r * M + c];
            }
            if (actual_e_col > S_COL + 2) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (actual_e_row > S_ROW + 2) std::cout << "  ..." << std::endl;
    }

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int64_t cpu_result = cpu_subarray_sum_2d(h_input, N, M, S_ROW, E_ROW, S_COL, E_COL);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " us" << std::endl;

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, total_size * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // GPU computation with benchmarking
    double gpu_time = benchmark(
        [&]() {
            solve(d_input, d_result, N, M, S_ROW, E_ROW, S_COL, E_COL);
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

    // Test various 2D array sizes and subarray ranges
    
    // Small arrays - full array
    test_subarray_sum_2d(4, 4, 0, 3, 0, 3, 100);           // 4x4, full
    test_subarray_sum_2d(8, 8, 0, 7, 0, 7, 100);           // 8x8, full
    
    // Small arrays - subarrays
    test_subarray_sum_2d(8, 8, 2, 5, 2, 5, 100);           // 8x8, inner 4x4
    test_subarray_sum_2d(8, 8, 0, 3, 4, 7, 100);           // 8x8, top-right quadrant
    test_subarray_sum_2d(8, 8, 4, 7, 0, 3, 100);           // 8x8, bottom-left quadrant
    
    // Medium arrays
    test_subarray_sum_2d(64, 64, 0, 63, 0, 63, 100);       // 64x64, full
    test_subarray_sum_2d(64, 64, 10, 50, 10, 50, 100);     // 64x64, inner subarray
    test_subarray_sum_2d(128, 256, 20, 100, 50, 200, 100); // Non-square, subarray
    
    // Large arrays
    test_subarray_sum_2d(512, 512, 0, 511, 0, 511, 100);         // 512x512, full
    test_subarray_sum_2d(512, 512, 100, 400, 100, 400, 100);     // 512x512, inner
    test_subarray_sum_2d(1024, 1024, 0, 1023, 0, 1023, 100);     // 1024x1024, full
    test_subarray_sum_2d(1024, 1024, 200, 800, 300, 700, 100);   // 1024x1024, inner
    
    // Very large arrays
    test_subarray_sum_2d(2048, 2048, 500, 1500, 500, 1500, 50);  // 2048x2048, inner 1001x1001
    
    // Non-square arrays
    test_subarray_sum_2d(100, 1000, 10, 90, 100, 900, 100);      // Wide array
    test_subarray_sum_2d(1000, 100, 100, 900, 10, 90, 100);      // Tall array
    
    // Edge cases - single row/column
    test_subarray_sum_2d(100, 100, 50, 50, 0, 99, 100);    // Single row
    test_subarray_sum_2d(100, 100, 0, 99, 50, 50, 100);    // Single column
    test_subarray_sum_2d(100, 100, 50, 50, 50, 50, 100);   // Single element
    
    // Corner subarrays
    test_subarray_sum_2d(100, 100, 0, 9, 0, 9, 100);       // Top-left corner
    test_subarray_sum_2d(100, 100, 0, 9, 90, 99, 100);     // Top-right corner
    test_subarray_sum_2d(100, 100, 90, 99, 0, 9, 100);     // Bottom-left corner
    test_subarray_sum_2d(100, 100, 90, 99, 90, 99, 100);   // Bottom-right corner
    
    std::cout << "\n============ All tests passed ===========\n" << std::endl;
    return 0;
}
