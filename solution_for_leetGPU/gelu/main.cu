#include "kernels.cuh"

// CPU reference implementation using exact GELU
void cpu_gelu_exact(const float* input, float* output, size_t N) {
    constexpr float SQRT_2_INV = 0.7071067811865475f;  // 1/sqrt(2)
    for (size_t i = 0; i < N; i++) {
        output[i] = input[i] * 0.5f * (1.0f + std::erf(input[i] * SQRT_2_INV));
    }
}

// CPU reference implementation using tanh approximation
void cpu_gelu_tanh(const float* input, float* output, size_t N) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    for (size_t i = 0; i < N; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

bool compare_results(const float* result, const float* expected, size_t N, float tolerance = 1e-5f) {
    for (size_t i = 0; i < N; i++) {
        float diff = std::abs(result[i] - expected[i]);
        float maxAbs = std::max(std::abs(result[i]), std::abs(expected[i]));
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
    
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    solve(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<float> result(N);
    CHECK_CUDA(cudaMemcpy(result.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
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
    
    free_device(d_input);
    free_device(d_output);
}

void test_case_auto(const std::vector<float>& input, const std::string& name) {
    size_t N = input.size();
    std::vector<float> expected(N);
    cpu_gelu_exact(input.data(), expected.data(), N);
    test_case(input, expected, name);
}

// External declarations for alternative solve functions
extern void solve_naive(const float* input, float* output, size_t N);
extern void solve_stride(const float* input, float* output, size_t N);
extern void solve_tanh(const float* input, float* output, size_t N);
extern void solve_unrolled(const float* input, float* output, size_t N);

void benchmark_test(size_t N) {
    std::cout << "\n=== Benchmark: N = " << N << " ===" << std::endl;
    
    float* h_input = alloc_host(N);
    rand_init(h_input, N);
    
    float* h_output_cpu = alloc_host(N);
    float* h_output_gpu = alloc_host(N);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gelu_exact(h_input, h_output_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    
    // Allocate device memory
    float* d_input = alloc_device(N);
    float* d_output = alloc_device(N);
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark different kernels
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU time:       " << cpu_time << " us" << std::endl;
    
    // Naive kernel
    double naive_time = benchmark([&]() {
        solve_naive(d_input, d_output, N);
    }, 100);
    std::cout << "Naive kernel:   " << naive_time << " us (speedup: " << cpu_time / naive_time << "x)" << std::endl;
    
    // Stride kernel
    double stride_time = benchmark([&]() {
        solve_stride(d_input, d_output, N);
    }, 100);
    std::cout << "Stride kernel:  " << stride_time << " us (speedup: " << cpu_time / stride_time << "x)" << std::endl;
    
    // Vectorized kernel (default solve)
    double vec_time = benchmark([&]() {
        solve(d_input, d_output, N);
    }, 100);
    std::cout << "Vectorized:     " << vec_time << " us (speedup: " << cpu_time / vec_time << "x)" << std::endl;
    
    // Tanh approximation kernel
    double tanh_time = benchmark([&]() {
        solve_tanh(d_input, d_output, N);
    }, 100);
    std::cout << "Tanh approx:    " << tanh_time << " us (speedup: " << cpu_time / tanh_time << "x)" << std::endl;
    
    // Unrolled kernel
    double unroll_time = benchmark([&]() {
        solve_unrolled(d_input, d_output, N);
    }, 100);
    std::cout << "Unrolled:       " << unroll_time << " us (speedup: " << cpu_time / unroll_time << "x)" << std::endl;
    
    // Verify result
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = compare_results(h_output_gpu, h_output_cpu, N, 1e-4f);
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    if (!correct) {
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
    
    free_device(d_input);
    free_device(d_output);
    free_host(h_input);
    free_host(h_output_cpu);
    free_host(h_output_gpu);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "=== GELU Activation Tests ===" << std::endl;
    std::cout << "GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))" << std::endl;
    std::cout << std::endl;
    
    // Test Example 1: Zero input -> 0
    // GELU(0) = 0 * 0.5 * (1 + erf(0)) = 0
    test_case({0.0f}, {0.0f}, "Zero input");
    
    // Test Example 2: Positive values
    // GELU approaches x for large positive x
    test_case_auto({1.0f, 2.0f, 3.0f}, "Positive values [1,2,3]");
    
    // Test Example 3: Negative values
    // GELU approaches 0 for large negative x
    test_case_auto({-1.0f, -2.0f, -3.0f}, "Negative values [-1,-2,-3]");
    
    // Test Example 4: Small values around zero
    test_case_auto({-0.5f, -0.1f, 0.1f, 0.5f}, "Small values [-0.5,-0.1,0.1,0.5]");
    
    // Test Example 5: Mixed values
    test_case_auto({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, "Mixed values [-2,-1,0,1,2]");
    
    // Test Example 6: Larger array
    std::vector<float> large_input(100);
    for (int i = 0; i < 100; i++) {
        large_input[i] = (i - 50) * 0.1f;  // Values from -5.0 to 4.9
    }
    test_case_auto(large_input, "100 elements");
    
    // Test Example 7: Single positive value
    // GELU(1.0) ≈ 0.8413
    test_case({1.0f}, {0.841345f}, "GELU(1.0)", 1e-4f);
    
    // Test Example 8: Single negative value  
    // GELU(-1.0) ≈ -0.1587
    test_case({-1.0f}, {-0.158655f}, "GELU(-1.0)", 1e-4f);
    
    // Benchmark tests with different sizes
    benchmark_test(1 << 10);    // 1K elements
    benchmark_test(1 << 14);    // 16K elements
    benchmark_test(1 << 16);    // 64K elements
    benchmark_test(1 << 18);    // 256K elements
    benchmark_test(1 << 20);    // 1M elements
    benchmark_test(1 << 22);    // 4M elements
    benchmark_test(1 << 24);    // 16M elements
    
    return 0;
}

